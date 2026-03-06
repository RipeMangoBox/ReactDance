import random
from math import log2
from functools import partial

from typing import List

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from models.nets.vector_quantizer.FSQ import FSQ

from einops import repeat, reduce, pack, unpack
from math import ceil
from einx import get_at
from concurrent.futures import ThreadPoolExecutor
from models.reactdance.layers.structure_utils import prob_mask_like
import torch.distributed as dist
from models.nets.nn import zero_module

# helper functions

def exists(val):
    return val is not None

def first(l):
    return l[0]

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# main class    
class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        levels: List[int],
        num_quantizers,
        channel_last = False,
        use_residual_norm = True,
        norm_type = "LayerNorm",
        quantize_dropout_prob = 1.0,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        soft_clamp_input_value = None,
        fsq_dropout_base_rate = 0.0,    #TODO
        r_rand_scale=0.0,               #TODO
        prior_mode="all_quantizeds",
        w_scale_dividison = True,
        **kwargs
    ):
        super().__init__()
        self.prior_mode = prior_mode
        self.codebook_dim = len(levels)
        dim = default(dim, self.codebook_dim)
        
        requires_projection = self.codebook_dim != dim
        self.project_in = nn.Linear(dim, self.codebook_dim) if requires_projection else nn.Identity()
        if requires_projection:
            if self.prior_mode in ["all_quantizeds", "bias_all_quantizeds"]:
                self.project_out = nn.Linear(self.codebook_dim, dim)
            elif self.prior_mode == "sep_all_quantizeds":
                self.project_out = nn.ModuleList([nn.Linear(self.codebook_dim, dim) for _ in range(num_quantizers)])
        else:
            self.project_out = nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers
        self.soft_clamp_input_value = soft_clamp_input_value
        
        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)
        self.channel_last = channel_last
        
        self.w_scale_dividison = w_scale_dividison
        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(
                levels = levels,
                dim = self.codebook_dim,
                channel_last = True, # because latents are transposed in rfsq if needed
                # preserve_symmetry=True,
                dropout_base_rate=fsq_dropout_base_rate * 0.5**ind,
                **kwargs["FSQ"]
            )

            self.layers.append(fsq)

        # assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer("scales", torch.stack(scales), persistent = False)
        
        self.quantize_dropout_prob = quantize_dropout_prob if num_quantizers > 1 else 0.0
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4
        
        # norm
        self.use_residual_norm = use_residual_norm
        self.norm = getattr(nn, norm_type)(self.codebook_dim) if self.use_residual_norm else None
        # print(f"\tResidualFSQ norm: {norm_type}") if self.use_residual_norm else print("\tNo Norm Used in ResidualFSQ")
        
        # metrics
        self.threshold = 1.0
        self.mu = 0.99
        self.codebook_usage = torch.ones(self.codebook_size, device=self.scales.device)
        self.r_rand_scale = r_rand_scale
        
        if self.training:
            self.initialize_weight()

    def initialize_weight(self):
        if self.has_projections:
            # print("\tResidual FSQ initialize_weight")
            self.project_in.apply(self._initialize_weight)
            # self.project_out.apply(self._initialize_weight)
            zero_module(self.project_out)
        
    def _initialize_weight(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks
        # 如果缓存已经计算过，直接返回缓存
        if self._cached_codebooks is None:
            codebooks = [layer.implicit_codebook for layer in self.layers]
            self._cached_codebooks = torch.stack(codebooks, dim=0)  # 缓存结果
        return self._cached_codebooks
    
    
    def _codes_to_embs(self, codes):
        # codes: [q b n l]
        # assert codes.shape[0] == self.num_quantizers and codes.shape[3] == self.codebook_dim """
        # scales = rearrange(self.scales, "q l -> q 1 1 l")
        scales = self.scales.unsqueeze(1).unsqueeze(1)
        return codes * scales

    def decode(self, zs, decode_mode="indices"):
        if decode_mode == "indices":
            output = self.indices_to_output(zs)
        # elif decode_mode == "quantizeds":
        #     output = self.project_out(zs)
        elif decode_mode == "all_embs":
            output = self.all_embs_to_output(zs)
        elif decode_mode in ["all_quantizeds", "sep_all_quantizeds", "bias_all_quantizeds"]:
            return zs.sum(dim=0)  # Sum all quantized values
        else:
            raise ValueError(f"decode_mode: {decode_mode} is not supported in ResidualFSQ")
        
        output_transpose = not self.channel_last
        if output_transpose:
            output = output.permute(0, 2, 1)  # Replaced 
        return output.float()
    
    def indices_to_all_embs_codes(self, indices):
        """
            old: b, n, q
            new: q, b, n
        """
        # quantize_dim = indices.shape[0]
        indices, ps = pack([indices], "q b *")

        # if quantize_dim < self.num_quantizers:
        #     assert self.quantize_dropout > 0., "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
        #     indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        mask = indices == -1
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = get_at("q [c] l, q b n -> q b n l", self.codebooks, indices) # self.codebooks [q, prod(levels), l], indices [b, n, q]
        all_codes = all_codes.masked_fill(mask.unsqueeze(-1), 0.)

        # scales = rearrange(self.scales, "q l -> q 1 1 l")
        scales = self.scales.unsqueeze(1).unsqueeze(1)
        all_embs = all_codes * scales
        all_embs, = unpack(all_embs, ps, "q b * l") # [q b n l]

        return all_embs, all_codes

    def indices_to_output(self, indices):
        all_embs, all_codes = self.indices_to_all_embs_codes(indices) # [q, b, n, l]
        output = self.decode(all_embs, decode_mode="all_embs")
        return output
    
    def all_embs_to_output(self, all_embs, need_sum=True):
        """
            old all_embs: b, n, q, l
            new all_embs: q, b, n, l
        """
        if self.prior_mode == "qunatizeds":
            # codes_summed = reduce(all_embs, "q b n l -> b n l", "sum") # b n q l -> b n l
            output = torch.sum(all_embs, dim=0) if need_sum else all_embs
            output = self.project_out(output)
        else:
            if self.prior_mode == "all_quantizeds":
                all_projected_embs = self.project_out(all_embs)
                # output = reduce(all_projected_embs, "q b n l -> b n l", "sum") # b n q l -> b n l
            elif self.prior_mode == "sep_all_quantizeds":
                all_projected_embs = torch.stack([self.project_out[q](all_embs[q]) for q in range(all_embs.size(0))], dim=0)
            elif self.prior_mode == "bias_all_quantizeds":   
                all_projected_embs = self.project_out(all_embs)  # (q, b, n, d_out)
                # 计算输入的均值和标准差 (沿b, n, d_in维度)
                mean_before = all_embs.mean(dim=[1, 2, 3])  # (q,)
                std_before = all_embs.std(dim=[1, 2, 3])    # (q,)

                # 计算输出的均值和标准差 (沿b, n, d_out维度)
                mean_after = all_projected_embs.mean(dim=[1, 2, 3])  # (q,)
                std_after = all_projected_embs.std(dim=[1, 2, 3])    # (q,)

                # 调整输出分布
                # 标准化并缩放到输入分布
                all_projected_embs = (all_projected_embs - mean_after[:, None, None, None]) / (std_after[:, None, None, None] + 1e-6)
                all_projected_embs = all_projected_embs * std_before[:, None, None, None] + mean_before[:, None, None, None]
                
            output = torch.sum(all_projected_embs, dim=0) if need_sum else all_projected_embs
        return output

    # @autocast(enabled = False)
    def encode_forward(
        self,
        x,
        encode_mode = "all_quantizeds",
    ):
        input_transpose = not self.channel_last
        if input_transpose:
            x = x.permute(0, 2, 1)

        """must disable autocast for the project_in layer, because amp will automatically cast the result to fp16 even the calculated result is in fp32"""
        with autocast(enabled=False):
            x = self.project_in(x)
            
        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value
        
        if self.use_residual_norm:
            x = self.norm(x)

        emb_out = 0.
        residual = x
        all_embs = []

        with autocast(enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                code = layer.encode_forward(residual / scale) if self.w_scale_dividison else layer.encode_forward(residual)
                emb = code * scale

                residual = residual - emb.detach()
                emb_out = emb_out + emb # not used

                all_embs.append(emb)
        
        # stack all indices
        all_embs = torch.stack(all_embs, dim = 0) # [q, b, n, l]

        if encode_mode == "quantizeds":
            if self.prior_mode in ["quantizeds", "all_quantizeds"]:
                output = self.project_out(emb_out)
            elif self.prior_mode in ["sep_all_quantizeds", "bias_all_quantizeds"]:
                output = self.all_embs_to_output(all_embs, need_sum=True)
        else:
            output = self.all_embs_to_output(all_embs, need_sum=False)

        return output
    
    # @autocast(enabled = False)
    def forward(
        self,
        x,
        output_transpose = True,
        no_metrics = False,
        rand_quantize_dropout_fixed_seed = None,
    ):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        input_transpose = not self.channel_last
        if input_transpose:
            x = x.permute(0, 2, 1)
        
        """must disable autocast for the project_in layer, because amp will automatically cast the result to fp16 even the calculated result is in fp32"""
        with autocast(enabled=False):
            x = self.project_in(x)
        # value_check(x, "x")
        
        # maybe softclamp input before residual layers
        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value
        
        if self.use_residual_norm:
            x = self.norm(x)

        emb_out = 0.
        residual = x
        # residual = first(self.layers).bound(x)  # this line use bound() of FSQ rather forward() of the first layer FSQ
        
        all_indices = []
        all_embs = []
        all_metrics = []
        
        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        if should_quantize_dropout:
            # check if seed is manually passed in
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)
                
            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indice = torch.full(x.shape[:2], -1., device = device, dtype = torch.long)
            # null_emb = torch.full(x.shape, 0., device = device, dtype = x.dtype)
            # null_single_commit_loss=torch.tensor(0., device=device)
            null_metric = dict(
                entropy=torch.tensor(1., device=device),    #TODO, think about the entropy
                used_curr=torch.tensor(1., device=device),
                usage=torch.tensor(1., device=device),
            )

        with autocast(enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indice)
                    null_emb = layer.indices_to_output(null_indice)
                    all_embs.append(null_emb)
                    all_metrics.append(null_metric)
                    continue

                code, indice, metric, single_commit_loss = layer(residual / scale, output_transpose=False, no_metrics=no_metrics) if self.w_scale_dividison else layer(residual, output_transpose=False, no_metrics=no_metrics)
                if quantizer_index == 0:
                    # code = code + self.r_rand_scale * torch.randn_like(code)
                    # code = (1 - self.r_rand_scale) * code + self.r_rand_scale * torch.randn_like(code)    # version 1
                    code += self.r_rand_scale * torch.randn_like(code)  # version2, only change the training polution, so it can fit both versions 
                emb = code * scale

                residual = residual - emb.detach()
                emb_out = emb_out + emb # not used

                all_indices.append(indice)
                all_embs.append(emb)
                if not no_metrics:
                    all_metrics.append(metric)
        
        # stack all indices
        all_indices = torch.stack(all_indices, dim = 0)
        all_embs = torch.stack(all_embs, dim = 0) # [q, b, n, l]
        all_metrics = {k: torch.stack([m[k] for m in all_metrics]) for k in all_metrics[0]} if not no_metrics else {}

        quantized_out = self.all_embs_to_output(all_embs, need_sum=True)
        
        if output_transpose:
            quantized_out = quantized_out.permute(0, 2, 1)  # Replaced

        ret = (quantized_out, all_indices, all_metrics)

        # value_check(quantized_out, "quantized_out")
        
        return ret

# grouped residual fsq

"""             
all_embs: [b, n, g, q, l], rearrange to [g, b, n, q, l] in decode of hfsq
[g, b, n, q, l]
    by (rfsq) project_out                   ->  [g, q, b, n, d] all_quantizeds
        by (rfsq) sum                       ->  [g, b, n, d]
        ->by (hfsq) cat                    ->  [b, n, g*d = D]
                -> by (hfsq) rearrange     ->  [b, D, n] quantized/output

indices -> all_codes
    -> all_embs (p_out)-> all_projected_embs (sum)-> [rfsq] quantized/output
        (sum)-> embs (p_out)-> [rfsq] quantized/output
            (cat, rearrange)-> [hfsq] quantized/output
            
normalizer keys: all_codes(skip), all_embs(done), embs(skip), quantizeds(done), all_quantizeds(to do)
supported decode_mode: indices, all_embs, embs, all_quantizeds, all_codes
"""
class HFSQ(Module):
    def __init__(
        self,
        *,
        dim = 128,
        groups = 8,
        accept_image_fmap = False,
        channel_last = True,
        use_group_norm = False,
        group_norm_type = "LayerNorm",
        batch_first = True,
        latent_loss_type = "commit",
        base_mask_rate = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rfsqs = nn.ModuleList([])

        for _ in range(groups):
            self.rfsqs.append(ResidualFSQ(
                dim = dim_per_group,
                channel_last=True, # because latents are transposed in hfsq if needed
                **kwargs
            ))

        # self.streams = [torch.cuda.Stream() for _ in range(groups)]  # 创建独立的 CUDA 流
        
        # norm
        self.use_group_norm = use_group_norm
        self.norm = getattr(nn, group_norm_type)(self.dim) if self.use_group_norm else None
        # print(f"Group norm: {group_norm_type}") if self.use_group_norm else print("No Norm Used in Group Level of GroupResidualFSQ")
        
        self._cached_codebooks = None
        self.codebook_size = self.rfsqs[0].codebook_size
        self.channel_last = channel_last
        self.batch_first = batch_first
        
        self.latent_loss_type = latent_loss_type
        self.base_mask_rate = base_mask_rate
        
    @property
    def codebooks(self):
        if self._cached_codebooks is None:
            self._cached_codebooks = torch.stack([rvq.codebooks for rvq in self.rfsqs])
        return self._cached_codebooks

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def indices_to_codes(self, indices):
        codes = tuple(rvq.indices_to_codes(chunk_indices) for rvq, chunk_indices in zip(self.rfsqs, indices))
        return torch.stack(codes)

    def indices_to_output(self, indices):
        outputs = tuple(rvq.indices_to_output(chunk_indices) for rvq, chunk_indices in zip(self.rfsqs, indices))    # tuple of group_num [b, n, d]
        outputs = torch.cat(outputs, dim = self.split_dim)
        
        ## emb in encode or forward of quantizer should be (b, n, d) to fit self.cond_emb in TemporalDownsampler of reactdance stage; but should be (b, d, n) in decode of quantizer to fit decode in encodec
        ## output in quantizers actually stands for emb
        output_transpose = not self.channel_last
        if output_transpose:
            output = outputs.permute(0, 2, 1)  # Replace
        return output  # [b, d, n]
    
    def _decode(self, zs, decode_mode="all_embs"):
        if decode_mode == "quantizer_quantizeds":
            # quantizer_quantizeds: [b, n, q, D]
            outputs = torch.sum(zs, dim=2)    # [b, n, D]
        elif decode_mode == "group_quantizeds":
            outputs = torch.cat(zs, dim = self.split_dim)
        elif decode_mode in ["all_quantizeds", "sep_all_quantizeds", "bias_all_quantizeds"]:
            outputs = tuple(rvq.decode(chunk_indices, decode_mode) for rvq, chunk_indices in zip(self.rfsqs, zs))    # group separation
            outputs = torch.cat(outputs, dim = self.split_dim)
        elif decode_mode == "quantizeds":
            outputs = zs
            
        ## emb in encode or forward of quantizer should be (b, n, d) to fit self.cond_emb in TemporalDownsampler of reactdance stage; but should be (b, d, n) in decode of quantizer to fit decode in encodec
        ## output in quantizers actually stands for emb
        output_transpose = not self.channel_last
        if output_transpose:
            output = outputs.permute(0, 2, 1)  # Replace
            
        return output.float()  # [b, d, n]

    def encode(self, x, encode_mode="all_quantizeds"):
        # return torch.randn(64, 60, 1024, device=x.device)
        # no norm used here because it"s already used in forward
        # quantized, all_embs, all_indices, all_metrics, all_codes, maybe_all_quantizeds = self.forward(x, output_transpose=False)
        
        # return torch.randn(64, 60, 1024, device=x.device), torch.randint(0, 999, [64, 8, 3, 60], device=x.device), torch.randn(64, 60, 8, 3, 128, device=x.device)
        
        output = self.encode_forward(x, encode_mode=encode_mode)
        return output.contiguous()
    
    def decode(self, zs, decode_mode="indices"):
        # assert decode_mode in ("indices", "embs", "all_embs", "all_quantizeds", "quantizer_quantizeds", "group_quantizeds"), f"decode_mode: {decode_mode} is not supported in HFSQ"
        
        if decode_mode == "indices":
            pass
            """new incides: g q b n"""
            # if self.batch_first:
            #     # zs = rearrange(zs, "b g q n -> g b n q")
            #     zs = zs.permute(1, 0, 3, 2)    # [b, g, n, q] -> [g, b, n, q]
        elif decode_mode in ["all_embs"]:
            pass
            """new all_embs: g q b n l"""
        elif decode_mode in ["all_quantizeds", "sep_all_quantizeds", "bias_all_quantizeds"]:
            if len(zs.shape) == 3:
                b, n, g, q = zs.size(0), zs.size(1), self.groups, self.rfsqs[0].num_quantizers
                zs = zs.view(b, n, g, q, -1)    # [b, n, g*q*l] -> [b, n, g, q, l]
            # zs = torch.unbind(rearrange(zs, "b n g q l -> g q b n l"), dim=0)   # g tuples of [b, n, q, l] codes  # unbind is not nesseary
            zs = zs.permute(2, 3, 0, 1, 4)   # g tuples of [b, n, q, l] codes  # more efficient
        # elif decode_mode == "group_quantizeds":
        #     # group_quantizeds: [b, n, g, d]
        #     # zs = torch.unbind(rearrange(zs, "b n g d -> g b n d"), dim=0)   # g tuples of [b, n, q, l] codes  # more efficient
        #     zs = zs.permute(2, 0, 1, 3)
        # return torch.randn(64, 1024, 60, device=zs.device)  # [b, d, n]
        return self._decode(zs, decode_mode)
    
    def encode_forward(
        self,
        x,
        encode_mode = "all_quantizeds",
    ):
        """
        所有的quantizer都是encode出来的quantizeds维度为(b,n,d), 是为了适应gen阶段的LAT中的`self.cond_emb`;
        forward和decode出来的quantizeds维度为(b,d,n), 是为了适配encodec的docoder。请注意emb的维度差异化处理是有意为之, 不是忘记统一！
        """
        
        input_transpose = not self.channel_last
        if input_transpose:
            x = x.permute(0, 2, 1)
                
        shape, split_dim = x.shape, self.split_dim
        # assert shape[split_dim] == self.dim """

        if self.use_group_norm:
            x = self.norm(x)
            
        # split the feature dimension into groups
        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            # rand_quantize_dropout_fixed_seed = random.randint(0, 1e7),
            encode_mode = encode_mode,
        )

        # invoke residual vq on each group
        out = tuple(rfsq.encode_forward(chunk, **forward_kwargs) for rfsq, chunk in zip(self.rfsqs, x))
        
        # otherwise, get all the zipped outputs and combine them
        if encode_mode == "quantizeds":
            output = torch.cat(out, dim = split_dim)
        elif encode_mode in ["all_quantizeds", "sep_all_quantizeds", "bias_all_quantizeds"]:
            output = torch.stack(out, dim=0)
            if self.batch_first:
                # all_quantizeds = rearrange(all_quantizeds, "g q b n d -> b n g q d")
                output = output.permute(2, 3, 0, 1, 4)    # [g, q, b, n, d] -> [b, n, g, q, d]
                
        return output
    
    # @autocast(enabled = False)
    def forward(
        self,
        x,
        output_transpose = True,
        no_metrics = False,
    ):
        """
        所有的quantizer都是encode出来的quantizeds维度为(b,n,d), 是为了适应gen阶段的LAT中的`self.cond_emb`;
        forward和decode出来的quantizeds维度为(b,d,n), 是为了适配encodec的docoder。请注意emb的维度差异化处理是有意为之, 不是忘记统一！
        """
        
        input_transpose = not self.channel_last
        if input_transpose:
            x = x.permute(0, 2, 1)
        input_latents = x.clone()
                
        shape, split_dim = x.shape, self.split_dim
        # assert shape[split_dim] == self.dim """

        if self.use_group_norm:
            x = self.norm(x)
            
        # split the feature dimension into groups
        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            # rand_quantize_dropout_fixed_seed = random.randint(0, 1e7),
            output_transpose = False,
            no_metrics = no_metrics,
        )

        # invoke residual vq on each group
        out = tuple(rfsq(chunk, **forward_kwargs) for rfsq, chunk in zip(self.rfsqs, x))
        out = tuple(zip(*out))
        
        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, all_metrics = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices, dim=0)
        all_metrics = {k: torch.stack([m[k] for m in all_metrics]) for k in all_metrics[0]} if not no_metrics else {}
        
        if self.latent_loss_type == "commit":
            commit_loss = F.mse_loss(input_latents, quantized.detach())
        elif self.latent_loss_type == "emb":
            commit_loss = F.mse_loss(input_latents.detach(), quantized)
        elif self.latent_loss_type == "commit_emb":
            commit_loss = F.mse_loss(input_latents, quantized.detach()) + F.mse_loss(input_latents.detach(), quantized)
       
        if self.base_mask_rate > 0:
            keep_mask_embed = prob_mask_like(shape, 1 - self.base_mask_rate, device=quantized.device)
            random_quantized = torch.randn_like(quantized)
            quantized = torch.where(keep_mask_embed, quantized, random_quantized)
        
        ## emb in encode or forward of quantizer should be (b, n, d) to fit self.cond_emb in TemporalDownsampler of reactdance stage; but should be (b, d, n) in decode of quantizer to fit decode in encodec
        ## output in quantizers actually stands for emb
        if output_transpose:
            quantized = quantized.permute(0, 2, 1)  # Replaced
        
        ret = (quantized, all_indices, commit_loss, all_metrics)
        return ret
    
def get_quantizer_quantizeds(all_quantizeds):
    if all_quantizeds is None or all_quantizeds[0] is None:
        return None
    # all_quantizeds: [b, n, g, q, d]
    # quantizer_quantizeds: [b, n, q, D]
    group_quantizeds = torch.split(all_quantizeds, 1, dim=2)    # [b, n, g, q, d] -> g * [b, n, 1, q, d]
    quantizer_quantizeds = torch.cat(group_quantizeds, dim=-1).squeeze(2)    # g * [b, n, 1, q, d] -> [b, n, q, D]
    return quantizer_quantizeds

def get_group_quantizeds(all_quantizeds):
    if all_quantizeds is None or all_quantizeds[0] is None:
        return None
    # all_quantizeds: [b, n, g, q, d]
    # group_quantizeds: [b, n, g, d]
    group_quantizeds = torch.sum(all_quantizeds, dim=-2)    # [b, n, g, q, d] -> g * [b, n, 1, q, d]
    return group_quantizeds


# region debug hfsq
"""             
all_embs: [b, n, g, q, l], rearrange to [g, b, n, q, l] in decode of hfsq
[g, b, n, q, l]
    by (rfsq) project_out                   ->  [g, q, b, n, d] all_quantizeds
        by (rfsq) sum                       ->  [g, b, n, d]
        ->by (hfsq) cat                    ->  [b, n, g*d = D]
                -> by (hfsq) rearrange     ->  [b, D, n] quantized/output

indices -> all_codes
    -> all_embs (p_out)-> all_projected_embs (sum)-> [rfsq] quantized/output
        (sum)-> embs (p_out)-> [rfsq] quantized/output
            (cat, rearrange)-> [hfsq] quantized/output
            
normalizer keys: all_codes(skip), all_embs(done), embs(skip), quantizeds(done), all_quantizeds(to do)
supported decode_mode: indices, all_embs, embs, all_quantizeds, all_codes
"""

def print_diff(tensor1, tensor2):
    # 找到不同的位置
    diff_mask = tensor1 != tensor2

    # 获取不同元素的位置
    indices = diff_mask.nonzero()

    # 打印不同位置的值
    values = []
    for i, j, k in indices:
        values.append(tensor1[i, j, k].item() - tensor2[i, j, k].item())
    print("max diff:", max(values) if len(values) else 0)        
        
def debug_grfsq_w_separate_project_out(levels):
    """
    b: batch
    d: dim
    n: n
    l: len(levels)
    g: group
    q: num_quantizers

    codes: [b, n, q, l]
    all_codes [g, b, n, q, l]
    embs: [b, n, q, l]
    
    indices: [b, g, q, n]
    all_embs: [b, n, g, q, l], converted from [g, b, n, q, l]
    all_quantizeds: [b, n, g, q, d], converted from [g, b, n, q, d]
    quantizer_quantizeds: [b, n, q, D]
    group_quantizeds: [b, n, g, d]
    output/quantizeds: [b, D, n] in forward and decode, but [b, n, D] in encode, which is an intentional operation!

    # name_check
    indices_to_codes: yes
    codes_to_indices: yes
    indices_to_output: yes
    """
    batch, dim, n = 351, 1024, 60
    hfsq = HFSQ(
        dim=dim,
        levels = levels,
        groups=8,
        num_quantizers = 3,
        channel_last=False,
        FSQ={"use_norm": True, "norm_type": "LayerNorm"},
    ).eval()
    
    usage = 0
    used_curr = 0
    for i in range(rounds):
        x = torch.randn(batch, dim, n) # (b, d, n), compatible with channel_last=False
        quantized, all_indices, all_metrics, maybe_all_quantizeds = hfsq(x) # (b, d, n), (g, b, n, q)
        # print(f"round {i}, usage: {metrics["usage"]}, cur_usage: {metrics["used_curr"]}, entropy: {metrics["entropy"]}")
        usage += all_metrics["usage"]
        used_curr += all_metrics["used_curr"]
    
    encoded_quantized, encoded_all_indices, maybe_all_quantizeds = hfsq.encode(x)
    decoded_from_all_quantizeds = hfsq.decode(maybe_all_quantizeds, decode_mode="all_quantizeds")

    assert torch.all(decoded_from_all_quantizeds == quantized)   # check all_quantizeds decode
    
    """Test3, Not Supported. do not run the following code, it"s just for a mark"""
    ## assert torch.all(encoded_quantized == quantized)  # Intentional differential treatment to fit the `decode` in encodec and `self.cond_emb` in TemporalDownsampler!
    
    return usage, used_curr
    
# endregion

if __name__ == "__main__":
    rounds = 3

    # debug_rfsq_usage()
    # debug_grfsq_wo_separate_project_out([8, 5, 5, 5, 5])
    debug_grfsq_w_separate_project_out([8, 5, 5, 5, 5])
    print("pass through")
