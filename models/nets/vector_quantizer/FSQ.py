"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from einops import pack, unpack, rearrange

# helper functions

eps = 1e-6

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float16, torch.float32, torch.float64),
        channel_last: bool = False,
        use_norm: bool = True,
        norm_type: str = 'LayerNorm',
        preserve_symmetry: bool = False,
        dropout_base_rate = 0.0,
        noise_type = "uniform",
        use_single_commit_loss: bool = False,
        **kwargs
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False) # to compress multiple levels indices to single codebook indices

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.quantize_noise_dropout = dropout_base_rate
        self.offset_noise_dropout = dropout_base_rate
        self.noise_type = noise_type

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        # assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_last = channel_last

        has_projections = use_single_commit_loss or self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()

        self.use_single_commit_loss = use_single_commit_loss
        self.has_projections = has_projections
        
        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        
        # norm
        self.use_norm = use_norm
        self.norm = getattr(nn, norm_type)(effective_codebook_dim) if self.use_norm else None
        # print(f'\t\tFSQ norm: {norm_type}') if self.use_norm else print('\t\tNo Norm Used in FSQ')

        # metrics
        self.threshold = 1.0
        self.mu = 0.99
        self.codebook_usage = torch.ones(self.codebook_size)
        
        if self.training:
            self.initialize_weight()

    def initialize_weight(self):
        if self.has_projections:
            # print('\t\tFSQ initialize_weight')
            self.project_in.apply(self._initialize_weight)
            self.project_out.apply(self._initialize_weight)
        
    def _initialize_weight(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels - 1)
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5
        return scale * bracket - 1.0
    
    def quantize(self, z: Tensor, preserve_symmetry=False):
        """Quantizes z, returns quantized zhat, same shape as z."""
        half_width = self._levels // 2
        unquantized = z  # 保留原始未量化值用于噪声处理

        if self.training:
            # ================= 第一阶段：基础量化 =================
            if preserve_symmetry:
                quantized = round_ste(self.symmetry_preserving_bound(z)) / half_width
            else:
                quantized = round_ste(self.bound(z)) / half_width
            
            # ================= 第二阶段：偏移噪声增强 =================
            # 生成偏移噪声掩码
            if self.offset_noise_dropout > eps:
                offset_mask = torch.bernoulli(
                    torch.full([z.shape[0], 1, 1, 1], 
                    self.offset_noise_dropout, 
                    device=z.device)
                ).bool().expand_as(z)
                
                # 生成符合量化动态范围的偏移量
                offset_noise = (torch.rand_like(z) * 2 - 1) if self.noise_type == "uniform" else torch.randn_like(z)
                offset = offset_noise * self._levels
                offset = self.bound(offset)  # 约束到有效量化范围
                
                # 应用偏移并重新量化（保持STE梯度）
                perturbed = self.bound(unquantized + offset)  # 偏移后的约束值
                offset_quantized = round_ste(perturbed) / half_width  # 重新离散化
                
                # 合并偏移结果
                quantized = torch.where(offset_mask, offset_quantized, quantized)

            # ================= 第三阶段：量化掩码替换 =================
            if self.quantize_noise_dropout > eps:
                quantize_mask = torch.bernoulli(
                    torch.full([z.shape[0], 1, 1, 1], 
                    self.quantize_noise_dropout, 
                    device=z.device)
                ).bool().expand_as(z)
                
                # 生成随机但符合量化约束的值（关键修改点）
                quantize_noise = (torch.rand_like(z) * 2 - 1) if self.noise_type == "uniform" else torch.randn_like(z)
                random_values = quantize_noise * self._levels  # [-L, L]范围
                random_quantized = round_ste(self.bound(random_values)) / half_width  # 完整量化流程
                
                # 应用量化掩码
                quantized = torch.where(quantize_mask, random_quantized, quantized)

        else:
            # 验证模式保持原始量化逻辑
            if preserve_symmetry:
                quantized = round_ste(self.symmetry_preserving_bound(z)) / half_width
            else:
                quantized = round_ste(self.bound(z)) / half_width
                
        return quantized
    
        """Quantizes z, returns quantized zhat (normalized to [-1, 1]), same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: Tensor):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat: Tensor):
        """Converts a `code` to an index in the codebook."""
        # assert zhat.shape[-1] == self.codebook_dim'''
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_output(self, indices: Tensor):
        # is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = codes.view(*codes.shape[:-2], -1)  # Flatten the last two dimensions (c, d) into one

        output = self.project_out(codes) # codes to output

        # if is_img_or_video or self.channel_first:
        #     output = rearrange(output, 'b ... d -> b d ...')
        
        ## emb in encode or forward of quantizer should be (b, n, d) to fit self.cond_emb in TemporalDownsampler of reactdance stage; but should be (b, d, n) in decode of quantizer to fit decode in encodec
        ## output in quantizers actually stands for emb
        # self.channel_last in fsq has been processed in init by rfsq
        output_transpose = not self.channel_last
        if output_transpose:
            output = output.permute(0, 2, 1)  # Replace 

        return output

    # not used actually, using forward instead
    # def encode(self, z):
    #     out, indices, metrics = self.forward(z, output_transpose=False, no_metrics=True)
    #     return indices, out
    
    def decode(self, indices: Tensor):
        return self.indices_to_output(indices)
    
    def calc_metrics(self, indices: Tensor):
        mu, dim = self.mu, self.codebook_size
        self.codebook_usage = self.codebook_usage.to(indices.device)
        with torch.no_grad():
            # Calculate new centres
            x_l_onehot = torch.zeros(dim, indices.shape[-1], device=indices.device)  # dim, N * L
            # assert indices.max() < dim, f"索引最大值 {indices.max()} 超出维度 {dim}"
            # assert indices.min() >= 0, "索引包含负值"
            x_l_onehot.scatter_(0, indices.to(torch.int64), 1)

            _k_elem = x_l_onehot.sum(dim=-1)  # dim

            # Update centres
            # self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, dim
            self.codebook_usage = mu * self.codebook_usage + (1. - mu) * _k_elem  # dim
            usage = (self.codebook_usage.view(dim, 1) >= self.threshold).float()
            # usage = (self.codebook_usage.view(dim, 1) >= 0).float()

            _k_prob = _k_elem / torch.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-6))  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = torch.sum(usage)

        return dict(entropy=entropy,
            used_curr=used_curr,
            usage=usage,
            )

    @autocast(enabled = False)
    def encode_forward(self, z: Tensor):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        orig_dtype = z.dtype
        
        input_transpose = not self.channel_last
        if input_transpose:
            z = z.permute(0, 2, 1)  # Replace 

        # assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}' '''

        z = self.project_in(z)

        z = z.view(z.shape[0], z.shape[1], self.num_codebooks, -1)  # Reshape to (b, n, c, d)

        # make sure allowed dtype before quantizing
        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        codes = self.quantize(z, preserve_symmetry=self.preserve_symmetry)
        codes = codes.view(codes.shape[0], codes.shape[1], -1)  # Flatten the last two dimensions (c, d) into one
        
        # cast codes back to original dtype
        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # project out
        out = self.project_out(codes)
        
        # return quantized output and indices
        return out
        
    @autocast(enabled = False)
    def forward(self, z: Tensor, output_transpose: bool = True, no_metrics=False):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        orig_dtype = z.dtype
        
        input_transpose = not self.channel_last
        if input_transpose:
            z = z.permute(0, 2, 1)  # Replace 

        # assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        if self.use_single_commit_loss:
            input_latents = z.clone()
            
        z = self.project_in(z)
        if self.use_norm:
            z = self.norm(z)

        z = z.view(z.shape[0], z.shape[1], self.num_codebooks, -1)  # Reshape to (b, n, c, d)

        # make sure allowed dtype before quantizing

        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        codes = self.quantize(z, preserve_symmetry=self.preserve_symmetry)
        indices = self.codes_to_indices(codes)

        codes = codes.view(codes.shape[0], codes.shape[1], -1)  # Flatten the last two dimensions (c, d) into one
        
        # metrics = self.calc_metrics(rearrange(indices, 'b d 1 -> 1 (b d)'))
        metrics = self.calc_metrics(indices.view(1, -1)) if not no_metrics else {}
        
        # cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes) 

        single_commit_loss = torch.tensor(0.0, device=z.device)
        if self.use_single_commit_loss:
            # loss shape (b, n, d)
            single_commit_loss = torch.mean(F.mse_loss(input_latents, out.detach(), reduction='none'), dim=(0, 1))
            
        if not self.keep_num_codebooks_dim:
            indices = indices.squeeze(-1)  # Replaced 

        ## emb in encode or forward of quantizer should be (b, n, d) to fit self.cond_emb in TemporalDownsampler of reactdance stage; but should be (b, d, n) in decode of quantizer to fit decode in encodec
        ## output in quantizers actually stands for emb
        if output_transpose:
            out = out.permute(0, 2, 1)  # Replaced 
        
        # return quantized output and indices
        return out, indices, metrics, single_commit_loss
    
'''
b: batch
d: dim
n: n
l: len(levels)

codes: [b, n, l]
indices: [b, n]
quantized/output: [b, d, n]

# name_check
indices_to_codes: yes
codes_to_indices: yes
indices_to_output: yes
'''

def fsq_test():
    rounds = 5
    for i in range(rounds):
        x = torch.randn(batch, dim, n).to(device) # (b, d, n)
        quantized, indices, metrics = fsq(x) # (b, d, n), (b, n)
    print('pass through')

if __name__ == '__main__':
    device = 'cuda'
    batch, dim, n = 256, 512, 60
    fsq = FSQ(
        dim=dim,
        levels = [8, 5, 5, 5],
        use_norm=True,
    ).to(device)

    fsq_test()    
    # fsq_encode_test()