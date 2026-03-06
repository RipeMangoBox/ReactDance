import torch
from torch import nn
import numpy as np
import math
from einops import rearrange

def value_check(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contain NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contain infinite values"

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def segment_pooling(x: torch.Tensor, segment_size: int = 60) -> torch.Tensor:
    """
    对 x 进行分段池化，每 segment_size 个时间步池化为一个特征向量。
    输入: x (b, seq_len, d)
    输出: (b, num_segments, d)
    """
    b, seq_len, d = x.shape
    num_segments = math.ceil(seq_len / segment_size)
    padded_len = num_segments * segment_size
    pad_size = padded_len - seq_len
    if pad_size > 0:
        x = torch.cat([x, torch.zeros(b, pad_size, d, device=x.device)], dim=1)
    x = x.view(b, num_segments, segment_size, d)
    x = x.mean(dim=2)  # (b, num_segments, d)
    return x

def expand_segments(segment_features: torch.Tensor, segment_size: int, original_seq_len: int) -> torch.Tensor:
    """
    将分段特征扩展回原始序列长度。
    输入: segment_features (b, num_segments, d)
    输出: (b, original_seq_len, d)
    """
    if segment_size == original_seq_len:
        return segment_features

    num_segments = math.ceil(original_seq_len / segment_size)
    expanded_sizes = [segment_size] * (num_segments - 1)
    expanded_sizes.append(original_seq_len - (num_segments - 1) * segment_size)
    
    # 向量化扩展
    expanded = torch.repeat_interleave(
        segment_features,
        torch.tensor(expanded_sizes, device=segment_features.device),
        dim=1
    )
    
    return expanded[:, :original_seq_len, :]

# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

    
def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k
# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, use_part_repeat_pe=False, num_parts=3, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.use_part_repeat_pe = use_part_repeat_pe
        
        assert num_parts == 3, "num_parts must be 3"
        self.num_parts = num_parts
        
        if self.use_part_repeat_pe:
            d_model = d_model // num_parts
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) # .transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, token_block_size = 60):
        """
        Args:
            x: Tensor
               - [bs, seq_len, d_model] (if use_part_repeat_pe=False)
               - [bs, joint_num, seq_len, d_model] (if use_part_repeat_pe=False)
               - [bs, seq_len, d_model_total] (if use_part_repeat_pe=True)
            token_block_size: 分块大小, default as 60, for motion length 240 frames
        Returns:
            output: Tensor (与x形状相同)
        """
        seq_len = x.size(-2)
        
        # 1. 计算基础PE，形状为 [seq_len, d_model]
        if token_block_size < 0:
            # Normal mode: Select first seq_len rows
            base_pe = self.pe[:seq_len, :]
        else:
            # 计算每个位置的块基索引
            block_base = (torch.arange(seq_len, device=x.device) % token_block_size)
            # 获取块基准位置编码
            # (注意：更高效的写法是 self.pe[block_base]，但保持你原有的逻辑)
            base_pe = self.pe[block_base] # [seq_len, d_model]
            
        # 2. 准备 final_pe
        # 添加 batch 维度
        final_pe = base_pe.unsqueeze(0)  # [1, seq_len, d_model]

        # 3. 根据 use_part_repeat_pe 应用不同逻辑
        if self.use_part_repeat_pe:
            # --- 模式二：重复PE以匹配 'up', 'down', 'tr' (假设 x 是 3D) ---
            
            if x.dim() != 3:
                raise ValueError(
                    f"use_part_repeat_pe=True 仅支持 3D 张量 (b, n, d_total), "
                    f"但收到了 {x.dim()}D 张量。"
                )
                
            # d_model_total = x.size(-1)
            # d_model_part = final_pe.size(-1)
            # if d_model_total % d_model_part != 0:
            #     raise ValueError(
            #         f"当 use_part_repeat_pe=True时, x 的最后一维 ({d_model_total}) "
            #         f"必须是 d_model ({d_model_part}) 的整数倍。"
            #     )
                
            num_parts = self.num_parts
            
            # 将 [1, seq_len, d_model] -> [1, seq_len, d_model * num_parts]
            final_pe = final_pe.repeat(1, 1, num_parts)

        else:
            # --- 模式一：原始逻辑 (concat 之前) ---
            if x.dim() == 4:  # [bs, joint_num, seq_len, d_model]
                # 添加一个维度以便广播到所有 joint
                final_pe = final_pe.unsqueeze(1)  # [1, 1, seq_len, d_model]
            
            # 如果 x.dim() == 3 [bs, seq_len, d_model],
            # final_pe [1, seq_len, d_model] 会自动广播
            
        # 4. 添加位置编码
        # - 模式一 (4D): x[b, 3, n, d] + final_pe[1, 1, n, d] -> OK
        # - 模式一 (3D): x[b, n, d]   + final_pe[1, n, d]   -> OK
        # - 模式二 (3D): x[b, n, 3*d] + final_pe[1, n, 3*d] -> OK
        x = x + final_pe 
        
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str,
                 out_dim = None, post_act_fn = None,
                 cond_proj_dim = None, zero_init_cond: bool = True) -> None:
        super(TimestepEmbedding, self).__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
            if zero_init_cond:
                self.cond_proj.weight.data.fill_(0.0)
        else:
            self.cond_proj = None

        # gelu
        self.act = torch.nn.GELU() if act_fn == 'gelu' else torch.nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = torch.nn.GELU() if post_act_fn == 'gelu' else torch.nn.SiLU()

    def forward(self, sample: torch.Tensor, timestep_cond = None) -> torch.Tensor:
        if timestep_cond is not None:
            sample = sample + self.cond_proj(timestep_cond)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class IdentityEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).unsqueeze(1)

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # (bs, seqlen, njoints, nfeats) -> (bs, njoints, seqlen, nfeats)
        x = x.permute((0, 2, 1, 3))
        x = self.poseEmbedding(x)
        return x
    
class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        # bs, n_joints, nframes, d = output.shape
        output = self.poseFinal(output) # (bs, n_joints, nframes, latent_dim) -> (bs, n_joints, nframes, input_feats)
        output = output.permute(0, 2, 1, 3) # (bs, n_joints, nframes, input_feats) -> (bs, nframes, n_joints, input_feats)
        bs, nframes, n_joints, input_feats = output.shape
        
        output = output.reshape(bs, nframes, -1) # (bs, nframes, n_joints, input_feats) -> (bs, nframes, n_joints * input_feats)
        return output


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool,
                 downscale_freq_shift: float) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift)
        return t_emb

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
