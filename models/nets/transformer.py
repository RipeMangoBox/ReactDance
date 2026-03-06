import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Final
from einops import rearrange
from .config import use_fused_attn
from .helpers import to_2tuple
from .nn import zero_module

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

def apply_rope(x, pos_emb, num_heads=None, output_reshape=True):
    # x shape: (batch_size * n, seq_len, dim) or (batch_size * seq_len, n_joints, dim)
    if num_heads is None:
        b, seq, num_heads, head_dim = x.shape
    else:
        b, seq, d = x.shape
    x = x.view(b, seq, num_heads, -1)
    x = x.permute(0, 2, 1, 3)  # (b, num_heads, seq, head_dim)
    x = x.reshape(b * num_heads, seq, -1)
    
    # Apply RoPE
    pos_emb = pos_emb(x)
    x, _ = apply_rotary_pos_emb(x, x, pos_emb)
    
    # Reshape back
    x = x.reshape(b, num_heads, seq, -1)
    x = x.permute(0, 2, 1, 3)  # (b, seq, num_heads, head_dim)
    if output_reshape:
        x = x.reshape(b, seq, -1)
    return x

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):  # Fixed method name with double underscores
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
class SelfAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            latent_dim,
            num_heads=8,
            attn_drop=0.1,
            proj_drop=0.1,
            qkv_bias=False,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            batch_first=True,
            use_attn_sigmod_gate=False,
            **kwargs
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, 'latent_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_sigmod_gate = nn.Sigmoid() if use_attn_sigmod_gate else nn.Identity()

        self.qkv = nn.Linear(latent_dim, latent_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Shape: (B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = self.attn_sigmod_gate(x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttentionROPE(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            latent_dim,
            num_heads=8,
            attn_drop=0.1,
            proj_drop=0.1,
            qkv_bias=False,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            batch_first=True,
            apply_rope=False,  # New flag to control RoPE application
            pos_emb=None,     # Positional embedding module for RoPE
            **kwargs
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, 'latent_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.apply_rope = apply_rope
        self.pos_emb = pos_emb

        self.qkv = nn.Linear(latent_dim, latent_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Shape: (B*seq_len, N, num_heads, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply RoPE to q and k if enabled
        if self.pos_emb is not None:
            if self.apply_rope:
                q = apply_rope(q.view(B, N, self.num_heads, self.head_dim), self.pos_emb, output_reshape=False)
                k = apply_rope(k.view(B, N, self.num_heads, self.head_dim), self.pos_emb, output_reshape=False)
            else:
                q = self.pos_emb(q)
                k = self.pos_emb(k)
            
        # (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            latent_dim,
            num_heads=8,
            attn_drop=0.1,
            proj_drop=0.1,
            qkv_bias=False,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            batch_first=True,
            use_attn_sigmod_gate=False,
            **kwargs
    ):
        super().__init__()
        assert latent_dim % num_heads == 0, 'latent_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.attn_sigmod_gate = nn.Sigmoid() if use_attn_sigmod_gate else nn.Identity()
        
        # Instead of a combined QKV projection, we have separate Q and KV projections
        self.q = nn.Linear(latent_dim, latent_dim, bias=qkv_bias)
        self.kv = nn.Linear(latent_dim, latent_dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(latent_dim, latent_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, attn_mask=None):
        """
        Args:
            x: Query input of shape (B, N, C)
            context: Key/Value input of shape (B, M, C)
        """
        B, N, C = x.shape
        M = context.shape[1]

        # Project queries from x
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values from context
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                # 扩展attn_mask到(1,1,N,N)，以便广播到(B, H, N, N)
                attn = attn.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = self.attn_sigmod_gate(x)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class JointAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=8, attn_drop=0.1, spatial_first=True, use_qknormsa=False, apply_rope=True, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        # RoPE embeddings for temporal and spatial dimensions
        self.temporal_pos = SinusoidalEmbeddings(self.head_dim)
        self.spatial_pos = SinusoidalEmbeddings(self.head_dim)
        
        if use_qknormsa:
            if apply_rope:
                self.temporal_attention = SelfAttentionROPE(latent_dim, num_heads, batch_first=True, dropout=attn_drop, pos_emb=self.temporal_pos)
                self.spatial_attention = SelfAttentionROPE(latent_dim, num_heads, batch_first=True, dropout=attn_drop, pos_emb=self.spatial_pos)
            else:
                self.temporal_attention = SelfAttention(latent_dim, num_heads, batch_first=True, dropout=attn_drop)
                self.spatial_attention = SelfAttention(latent_dim, num_heads, batch_first=True, dropout=attn_drop)
        else:
            self.temporal_attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True, dropout=attn_drop)
            self.spatial_attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True, dropout=attn_drop)
            
        self.spatial_first = spatial_first
        self.use_qknormsa = use_qknormsa
    
    # def _apply_rope(self, x, pos_emb):
    #     # x shape: (batch_size * n, seq_len, dim) or (batch_size * seq_len, n_joints, dim)
    #     b, seq, d = x.shape
    #     x = x.view(b, seq, self.num_heads, -1)
    #     x = x.permute(0, 2, 1, 3)  # (b, num_heads, seq, head_dim)
    #     x = x.reshape(b * self.num_heads, seq, -1)
        
    #     # Apply RoPE
    #     pos_emb = pos_emb(x)
    #     x, _ = apply_rotary_pos_emb(x, x, pos_emb)
        
    #     # Reshape back
    #     x = x.reshape(b, self.num_heads, seq, -1)
    #     x = x.permute(0, 2, 1, 3)  # (b, seq, num_heads, head_dim)
    #     x = x.reshape(b, seq, -1)
    #     return x
    
    def _apply_temporal_attention(self, x, temporal_mask=None):
        b, seq_len, n, dim = x.shape
        temp_x = x.permute(0, 2, 1, 3).reshape(b * n, seq_len, dim)
        
        # Apply temporal attention
        if not self.use_qknormsa:
            # Apply RoPE
            temp_x = apply_rope(temp_x, self.temporal_pos, num_heads=self.num_heads, output_reshape=True)
            temporal_out, _ = self.temporal_attention(temp_x, temp_x, temp_x, attn_mask=temporal_mask)
        else:
            temporal_out = self.temporal_attention(temp_x, attn_mask=temporal_mask)
            
        temporal_out = temporal_out + temp_x
        return temporal_out.reshape(b, seq_len, n, dim).permute(0, 2, 1, 3)
    
    def _apply_spatial_attention(self, x, spatial_mask=None):
        b, seq_len, n_joints, dim = x.shape
        spatial_x = x.reshape(b * seq_len, n_joints, dim)
        
        # Apply spatial attention
        if not self.use_qknormsa:
            # Apply RoPE
            spatial_x = apply_rope(spatial_x, self.spatial_pos, num_heads=self.num_heads, output_reshape=True)
            spatial_out, _ = self.spatial_attention(spatial_x, spatial_x, spatial_x, attn_mask=spatial_mask)
        else:
            spatial_out = self.spatial_attention(spatial_x, attn_mask=spatial_mask)
        
        spatial_out = spatial_out + spatial_x
        return spatial_out.reshape(b, seq_len, n_joints, dim)
        
    def forward(self, x, joint_num, temporal_mask=None, spatial_mask=None, output_reshape=True):
        if len(x.shape) == 3:
            b, seq_len, D = x.shape
            dim = D // joint_num
            x = x.view(b, seq_len, joint_num, dim)
        else:
            b, joint_num, seq_len, dim = x.shape
            x = x.permute(0, 2, 1, 3)
            D = joint_num * dim
        
        if self.spatial_first:
            x = self._apply_spatial_attention(x, spatial_mask=spatial_mask)
            x = self._apply_temporal_attention(x, temporal_mask=temporal_mask)
        else:
            x = self._apply_temporal_attention(x, temporal_mask=temporal_mask)
            x = self._apply_spatial_attention(x, spatial_mask=spatial_mask)
            
        if output_reshape:
            return x.reshape(b, seq_len, D)
        else:
            return x
     
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.1,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            latent_dim=512,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=True,
            proj_drop=0.1,
            attn_drop=0.1,
            init_values=None,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            use_attn_sigmod_gate=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(latent_dim)
        self.attn = SelfAttention(
            latent_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_attn_sigmod_gate=use_attn_sigmod_gate,
        )
        self.ls1 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(latent_dim)
        self.mlp = mlp_layer(
            in_features=latent_dim,
            hidden_features=int(latent_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, temporal_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), temporal_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            latent_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=True,
            proj_drop=0.1,
            attn_drop=0.1,
            init_values=None,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            use_attn_sigmod_gate=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(latent_dim)
        
        self.cross_attn = CrossAttention(
            latent_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_attn_sigmod_gate=use_attn_sigmod_gate,
        )
        self.ls1 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(latent_dim)
        self.mlp = mlp_layer(
            in_features=latent_dim,
            hidden_features=int(latent_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context, temporal_mask=None):
        x = x + self.drop_path1(self.ls1(
            self.cross_attn(self.norm1(x), context, temporal_mask)
        ))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class SpatialTemporalBlock(nn.Module):
    def __init__(
            self,
            latent_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=True,
            proj_drop=0.1,
            attn_drop=0.1,
            init_values=None,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            spatial_first=True,
            apply_rope=True,
            use_qknormsa=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(latent_dim)
       
        self.spatial_temporal_attn = JointAttention(latent_dim, num_heads=num_heads, attn_drop=attn_drop, spatial_first=spatial_first, use_qknormsa=(qk_norm and use_qknormsa), apply_rope=apply_rope)
        self.ls1 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(latent_dim)
        self.mlp = mlp_layer(
            in_features=latent_dim,
            hidden_features=int(latent_dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(latent_dim)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls3 = LayerScale(latent_dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, joint_num, temporal_mask=None):
       # apply spatial, then temporal attention
        x = x + self.drop_path1(self.ls1(self.spatial_temporal_attn(self.norm2(x), joint_num, temporal_mask=temporal_mask, output_reshape=False)))

        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x

class AdaLN(nn.Module):
    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, h, emb=None):
        """
        h: B, T, D
        emb: B, D or None
        """
        if emb is None:
            # 如果没有提供emb，则直接返回归一化后的h
            return self.norm(h)
        else:
            # B, 1, 2D
            # emb_out = self.emb_layers(emb)
            # # scale: B, 1, D / shift: B, 1, D
            # scale, shift = torch.chunk(emb_out, 2, dim=-1)
            # h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
            # return h
        
            # 检查 emb 的形状
            if emb.dim() == 2:  # emb 形状为 [B, D]
                emb = emb.unsqueeze(1)  # 扩展为 [B, 1, D]
            elif emb.dim() != 3:  # emb 形状不合法
                raise ValueError("emb must be of shape [B, D] or [B, T, D]")
            
            # 处理 emb_out
            emb_out = self.emb_layers(emb)  # 输出形状为 [B, T, 2D]
            scale, shift = torch.chunk(emb_out, 2, dim=-1)  # 拆分为 [B, T, D]

            # 归一化并应用 scale 和 shift
            h = self.norm(h) * (1 + scale) + shift
            return h

# class VanillaSelfAttention(nn.Module):
#     def __init__(self, latent_dim, num_head, attn_drop=0., embed_dim=None, batch_first=True, **kwargs):
#         super().__init__()
#         self.num_head = num_head
#         self.norm = AdaLN(latent_dim, embed_dim)
#         self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=attn_drop, batch_first=batch_first,
#                                                add_zero_attn=False)
        
#     def forward(self, x, emb, temporal_mask=None):
#         """
#         x: B, T, D
#         """
#         x_norm = self.norm(x, emb)
#         y = self.attention(x_norm, x_norm, x_norm,
#                            attn_mask=temporal_mask,
#                            need_weights=False)[0]
#         return y

# class VanillaCrossAttention(nn.Module):
#     def __init__(self, latent_dim, num_head, attn_drop=0., embed_dim=None, batch_first=True, **kwargs):
#         super().__init__()
#         self.num_head = num_head
#         self.norm = AdaLN(latent_dim, embed_dim)
#         self.xf_norm = AdaLN(latent_dim, embed_dim)
#         self.attention = nn.MultiheadAttention(latent_dim, num_head, kdim=latent_dim, vdim=latent_dim,
#                                                dropout=attn_drop, batch_first=batch_first, add_zero_attn=False)
        
#     def forward(self, x, xf, diffusion_emb, diffusion_music_emb, temporal_mask=None):
#         """
#         x: B, T, D
#         xf: B, N, L
#         """
#         x_norm = self.norm(x, diffusion_music_emb)
#         xf_norm = self.xf_norm(xf, diffusion_emb)
#         y = self.attention(x_norm, xf_norm, xf_norm, # q, k, v
#                            attn_mask=temporal_mask,
#                            need_weights=False)[0]
#         return y

class FFN(nn.Module):
    def __init__(self, latent_dim, mlp_ratio, dropout, embed_dim=None):
        super().__init__()
        ffn_dim = int(latent_dim * mlp_ratio)
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        x_norm = self.norm(x, emb) if emb is not None else x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y

class FinalLayer(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.linear = zero_module(nn.Linear(latent_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.linear(x)
        return x
    
# class TransformerBlock_sca(nn.Module):
#     def __init__(self,
#                  latent_dim=512,
#                  num_heads=8,
#                  mlp_ratio=4,
#                  attn_drop=0.1,
#                  FFN_diffusion_emb_only=False,
#                  sa_backend_name="VANILLASA",
#                  ca_backend_name="VANILLACA",
#                  **kargs):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.num_heads = num_heads
#         self.FFN_diffusion_emb_only = FFN_diffusion_emb_only
        
#         self.sa_block = VanillaSelfAttention(latent_dim, num_heads, attn_drop)
#         self.ca_block = VanillaCrossAttention(latent_dim, num_heads, attn_drop, latent_dim)
#         self.ffn = FFN(latent_dim, mlp_ratio, attn_drop, latent_dim)

#     def forward(self, x, cond_emb, diffusion_emb, music_emb, temporal_mask=None):
#         diffusion_music_emb = diffusion_emb + music_emb if music_emb is not None else diffusion_emb
#         ffn_emb = diffusion_emb if self.FFN_diffusion_emb_only else diffusion_music_emb
        
#         h1 = self.sa_block(x, diffusion_music_emb, temporal_mask)
#         h1 = h1 + x
#         h2 = self.ca_block(h1, cond_emb, diffusion_emb, diffusion_music_emb, temporal_mask)
#         h2 = h2 + h1
#         out = self.ffn(h2, ffn_emb)   #attention, not determined to use which emb for ffn
#         out = out + h2

#         return out

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self,
#                  latent_dim=512,
#                  num_heads=8,
#                  ff_size=1024,
#                  attn_drop=0.1,
#                  attn_backend_name="VANILLASA",
#                  **kargs):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.num_heads = num_heads

#         self.sa_block = VanillaSelfAttention(latent_dim, num_heads, attn_drop)
#         self.ffn = FFN(latent_dim, ff_size, attn_drop, latent_dim)

#     def forward(self, x, emb=None, temporal_mask=None):
#         h1 = self.sa_block(x, emb, temporal_mask)
#         h1 = h1 + x
#         out = self.ffn(h1, emb)
#         out = out + h1

#         return out
    
def get_attn_backend(backend_name: str):
    # if backend_name == "MHA":
    #     return nn.MultiheadAttention
    # elif backend_name == "VANILLASA":
    #     return VanillaSelfAttention
    # elif backend_name == "VANILLACA":
    #     return VanillaCrossAttention
    if backend_name == "QKNORMSA":
        return SelfAttention
    elif backend_name == "QKNORMCA":
        return CrossAttention
    # elif backend_name == "ROPEQKSA":
    #     return SelfAttentionROPE
    # elif backend_name == "JOINTATTN":
    #     return JointAttention
    else:
        raise ValueError(f"Unknown attention backend: {backend_name}")

def get_attn_block_backend(block_backend_name: str):
    # if block_backend_name == "TransformerEncoderBlock":
    #     return TransformerEncoderBlock
    # elif block_backend_name == "TransformerBlock_sca":
    #     return TransformerBlock_sca
    if block_backend_name == "SelfAttentionBlock":
        return SelfAttentionBlock
    elif block_backend_name == "CrossAttentionBlock":
        return CrossAttentionBlock
    # elif block_backend_name == "SpatialTemporalBlock":
    #     return SpatialTemporalBlock
    else:
        raise ValueError(f"Unknown attention block backend: {block_backend_name}")