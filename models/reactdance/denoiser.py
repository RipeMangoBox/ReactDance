from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from models.nets.nn import FeatureEncoder
from models.reactdance.layers.rotary_embedding_torch import RotaryEmbedding
from models.reactdance.layers.structure_utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
import math
from models.nets.transformer import get_attn_backend, AdaLN

def segment_pooling(x: Tensor, segment_size: int = 60) -> Tensor:
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

def expand_segments(segment_features: Tensor, segment_size: int, original_seq_len: int) -> Tensor:
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

def blockwise_pooling(x: Tensor, segment_size: int = 60) -> Tensor:
    seqlen = x.size(1)
    if segment_size == 1:
        return x
    elif segment_size == -1 or seqlen == segment_size:
        return x.mean(-2, keepdim=True)
    
    segment_features = segment_pooling(x, segment_size)  # (b, num_segments, latent_dim)
    cond_blockwise_hidden = expand_segments(segment_features, segment_size, x.size(1))  # (b, seq_len, latent_dim)
    
    return cond_blockwise_hidden
    
class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        # pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        dtype=None,
        rotary=None,
        sa_backend="MHA",
        use_attn_sigmod_gate=False,
    ) -> None:
        super().__init__()
        self.sa_backend = sa_backend
        sa_backend = get_attn_backend(sa_backend)
        self.self_attn = sa_backend(
            d_model, nhead, dropout, batch_first=batch_first, use_attn_sigmod_gate=use_attn_sigmod_gate,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, temporal_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        if self.sa_backend == "MHA":
            x = self.self_attn(
                qk,
                qk,
                x,
                attn_mask=temporal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            x = self.self_attn(
                x,
                attn_mask=temporal_mask,
            )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        dtype=None,
        rotary=None,
        sa_backend="MHA",
        ca_backend="MHA",
        use_adaln=False,
        use_attn_sigmod_gate=False,
    ):
        super().__init__()
        self.sa_backend = sa_backend
        sa_backend = get_attn_backend(sa_backend)
        self.self_attn = sa_backend(
            d_model, nhead, dropout=dropout, batch_first=batch_first, use_attn_sigmod_gate=use_attn_sigmod_gate,
        )
        
        self.ca_backend = ca_backend
        ca_backend = get_attn_backend(ca_backend)
        self.multihead_attn = ca_backend(
            d_model, nhead, dropout=dropout, batch_first=batch_first, use_attn_sigmod_gate=use_attn_sigmod_gate,
        )
        self.use_adaln = use_adaln
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        if self.use_adaln:
            self.norm1 = AdaLN(d_model)
            self.norm2 = AdaLN(d_model)
            self.norm3 = AdaLN(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation
        
        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            if self.use_adaln:
                # self-attention -> residual
                x = x + self._sa_block(self.norm1(x, t), tgt_mask, tgt_key_padding_mask)

                # cross-attention -> residual
                x = x + self._mha_block(
                    self.norm2(x, t), memory, memory_mask, memory_key_padding_mask
                )

                # feedforward -> residual
                x = x + self._ff_block(self.norm3(x, t))
            else:
                # self-attention -> film -> residual
                x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
                x = x + featurewise_affine(x_1, self.film1(t))
                # cross-attention -> film -> residual
                x_2 = self._mha_block(
                    self.norm2(x), memory, memory_mask, memory_key_padding_mask
                )
                x = x + featurewise_affine(x_2, self.film2(t))
                # feedforward -> film -> residual
                x_3 = self._ff_block(self.norm3(x))
                x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, temporal_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        if self.sa_backend == "MHA":
            x = self.self_attn(
                qk,
                qk,
                x,
                attn_mask=temporal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            x = self.self_attn(
                x,
                attn_mask=temporal_mask,
            )
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, temporal_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        if self.ca_backend == "MHA":
            x = self.multihead_attn(
                q,
                k,
                mem,
                attn_mask=temporal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            x = self.multihead_attn(
                x,
                mem,
                attn_mask=temporal_mask,
            )
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,):
        for layer in self.stack:
            x = layer(x, cond, t, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return x

class ReactDance_Denoiser(nn.Module):#feed pose cond/uncond; feed music cond/uncond; split
    def __init__(
        self,
        input_feats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        n_actor=2,      # leader, music
        music_time_merged=False,
        config=None,
        **kwargs
    ) -> None:

        super().__init__()

        LATENT_DIM = getattr(config, "latent_dim", 512)
        print(f"[LATENT_DIM: {LATENT_DIM}]")
        
        self.n_actor = n_actor
        self.n_reactor = 3  # (up, down, tr)
        self.music_time_merged = music_time_merged  # if True, set cond_tokens to p_cond_tokens and cond_hiddens to m_cond_hidden
        output_feats = np.prod(getattr(config, "pos_allq_shape", [int(self.n_reactor * input_feats)]))
        self.feature_encoder = FeatureEncoder(config=config)
        encode_norm_first = getattr(config, "encode_norm_first", True)
        decode_norm_first = getattr(config, "decode_norm_first", True)
        use_adaln = getattr(config, "use_adaln", False)
        sa_backend = getattr(config.gpt, "sa_backend", "MHA")
        ca_backend = getattr(config.gpt, "ca_backend", "MHA")
        use_attn_sigmod_gate = getattr(config.gpt, 'use_attn_sigmod_gate', False)
        self.guidance_drop_mode = getattr(config, "guidance_drop_mode", "learnable")
        
        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        use_rotary = getattr(config, "use_rotary", False)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=LATENT_DIM)
        else:
            # we can use part-repeat PE because the input_projection is an identity
            use_part_repeat_pe = getattr(config, "use_part_repeat_pe", False)
            self.abs_pos_encoding = PositionalEncoding(
                LATENT_DIM, use_part_repeat_pe, self.n_reactor, dropout
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(LATENT_DIM),  # learned?
            nn.Linear(LATENT_DIM, LATENT_DIM * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(LATENT_DIM * 4, LATENT_DIM),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(LATENT_DIM * 4, LATENT_DIM * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        ## null embeddings for guidance dropout
        self.null_m_cond_embed = nn.Parameter(torch.randn(1, 1, LATENT_DIM) * 0.02)
        self.null_m_cond_hidden = nn.Parameter(torch.randn(1, LATENT_DIM) * 0.02)

        self.null_p_cond_embed = nn.Parameter(torch.randn(1, 1, LATENT_DIM) * 0.02)
        self.null_p_cond_hidden = nn.Parameter(torch.randn(1, LATENT_DIM) * 0.02)

        self.norm_cond = nn.LayerNorm(LATENT_DIM)

        # input projection
        self.input_projection = nn.Identity() if output_feats == LATENT_DIM else nn.Linear(output_feats, LATENT_DIM)
        self.p_cond_encoder = nn.Sequential()
        self.m_cond_encoder = nn.Sequential()
        for _ in range(2):
            self.p_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=LATENT_DIM,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=encode_norm_first,
                    rotary=self.rotary,
                    sa_backend=sa_backend,
                    use_attn_sigmod_gate=use_attn_sigmod_gate,
                )
            )
            self.m_cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=LATENT_DIM,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=encode_norm_first,
                    rotary=self.rotary,
                    sa_backend=sa_backend,
                    use_attn_sigmod_gate=use_attn_sigmod_gate,
                )
            )
        # conditional projection
        # self.p_cond_projection = nn.Linear(self.p_cond_dim, LATENT_DIM)  # 4096
        # self.m_cond_projection = nn.Linear(self.m_cond_dim, LATENT_DIM)  # 1024
        LAT_force_dim = getattr(config, "LAT_force_dim", False)
        self.p_cond_projection = nn.Identity() if LAT_force_dim or output_feats == LATENT_DIM else nn.Linear(output_feats, LATENT_DIM)
        self.m_cond_projection = nn.Identity() if LAT_force_dim or output_feats == LATENT_DIM else nn.Linear(output_feats, LATENT_DIM)
        self.non_attn_p_cond_projection = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.SiLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
        )
        self.non_attn_m_cond_projection = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.SiLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    LATENT_DIM,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=decode_norm_first,
                    rotary=self.rotary,
                    sa_backend=sa_backend,
                    ca_backend=ca_backend,
                    use_adaln=use_adaln,
                )
            )

        if not music_time_merged:
            self.fuse_pm_tokens = nn.Linear(LATENT_DIM * 2, LATENT_DIM)
            self.fuse_pm_hidden = nn.Linear(LATENT_DIM * 2, LATENT_DIM)
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Identity() if output_feats == LATENT_DIM else nn.Linear(LATENT_DIM, output_feats)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def guided_forward(self, x, cond_tuple, timesteps, pm_guidance_weight, levels=[], ignore_guide=False, body_parts=['up', 'down', 'tr'], is_last_step=False, **model_kwargs):
        '''
        levels can be used to apply guidance on selected scales independently
        '''
        if ignore_guide:
            return self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=0, m_cond_drop_prob=0, **model_kwargs)
        
        # unc = self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=1, m_cond_drop_prob=1, **model_kwargs)
        # p_conditioned = self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=0, m_cond_drop_prob=1, **model_kwargs)
        # pm_conditioned = self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=0, m_cond_drop_prob=0, **model_kwargs)

        # output = unc + (p_conditioned - unc) * m_guidance_weight + (pm_conditioned - p_conditioned) * pm_guidance_weight
        # return output
        unc = self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=1, m_cond_drop_prob=1, **model_kwargs)
        pm_conditioned = self.forward(x, cond_tuple, timesteps, p_cond_drop_prob=0, m_cond_drop_prob=0, **model_kwargs)

        return pm_conditioned

        if is_last_step:
            unc_up, unc_down, unc_tr = torch.split(unc, unc.shape[-1]//3, dim=-1)
            pm_up, pm_down, pm_tr = torch.split(pm_conditioned, pm_conditioned.shape[-1]//3, dim=-1)
            
            up_factor = m_guidance_weight if 'up' in body_parts else pm_guidance_weight
            down_factor = m_guidance_weight if 'down' in body_parts else pm_guidance_weight
            tr_factor = m_guidance_weight if 'tr' in body_parts else pm_guidance_weight
            
            output_up = unc_up + (pm_up - unc_up) * up_factor
            output_down = unc_down + (pm_down - unc_down) * down_factor
            output_tr = unc_tr + (pm_tr - unc_tr) * tr_factor
                
            output = torch.cat([output_up, output_down, output_tr], dim=-1)
        else:
            output = unc + (pm_conditioned - unc) * pm_guidance_weight

        output = output.view(output.shape[0], output.shape[1], -1)
        return output
    
    #     # return pm_conditioned + (pm_conditioned - p_conditioned) * m_guidance_weight + (p_conditioned - unc) * pm_guidance_weight
    #     b, n, D = pm_conditioned.shape
    #     ret = pm_conditioned + (pm_conditioned - unc) * m_guidance_weight
    #     if len(levels):
    #         ret = ret.view(b, n, 3, D//3)
    #         pm_conditioned = pm_conditioned.view(b, n, 3, D//3)
    #         # p_conditioned = p_conditioned.view(b, n, 3, D//3)
    #         unc = unc.view(b, n, 3, D//3)
            
    #         pm_selected_scale = pm_conditioned[:, :, levels, :]
    #         unc_selected_scale = unc[:, :, levels, :]
    #         ret[:, :, levels, :] = pm_selected_scale + (pm_selected_scale - unc_selected_scale) * m_guidance_weight
    #         ret = ret.view(b, n, -1)
        
    #     return ret

    def forward(
        self, 
        x: Tensor,
        cond_tuple: Tensor,
        timesteps: Tensor,
        mask: Tensor,
        p_cond_drop_prob: float = 0.0,
        m_cond_drop_prob: float = 0.0,
        token_block_size: int = 60,
        **kwargs,
    ):
        cond = self.feature_encoder(cond_tuple, joint_num=3, attn_mask=mask)
        m_cond, p_cond = torch.split(cond, cond.shape[-1]//2, dim=-1)
        batch_size, seqlen, device = x.shape[0], x.shape[1], x.device

        # Project to latent space
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x, token_block_size)
        
        # --- Process Pose (p) and Music (m) Conditions ---
        p_cond_tokens = self.p_cond_encoder(self.abs_pos_encoding(self.p_cond_projection(p_cond), token_block_size))
        m_cond_tokens = self.m_cond_encoder(self.abs_pos_encoding(self.m_cond_projection(m_cond), token_block_size))
        
        p_cond_blockwise_hidden = blockwise_pooling(p_cond_tokens, token_block_size)
        p_cond_hidden = self.non_attn_p_cond_projection(p_cond_blockwise_hidden)
        
        m_cond_blockwise_hidden = blockwise_pooling(m_cond_tokens, token_block_size)
        m_cond_hidden = self.non_attn_m_cond_projection(m_cond_blockwise_hidden)

        # --- Apply Guidance Dropout based on self.guidance_drop_mode ---
        
        # 1. Create dropout masks (True = keep, False = drop)
        p_keep_mask = prob_mask_like((batch_size,), 1 - p_cond_drop_prob, device=device)
        m_keep_mask = prob_mask_like((batch_size,), 1 - m_cond_drop_prob, device=device)

        if self.guidance_drop_mode == 'learnable':
            # Reshape masks for broadcasting
            p_keep_mask_embed = p_keep_mask.view(batch_size, 1, 1)
            p_keep_mask_hidden = p_keep_mask.view(batch_size, 1, 1) # Note: shape corrected for broadcasting with (bs, 1, dim)
            m_keep_mask_embed = m_keep_mask.view(batch_size, 1, 1)
            m_keep_mask_hidden = m_keep_mask.view(batch_size, 1, 1) # Note: shape corrected
            
            # Replace dropped conditions with learnable null embeddings
            null_p_cond_embed = self.null_p_cond_embed.to(p_cond_tokens.dtype).repeat(batch_size, seqlen, 1)
            p_cond_tokens = torch.where(p_keep_mask_embed, p_cond_tokens, null_p_cond_embed)

            null_m_cond_embed = self.null_m_cond_embed.to(m_cond_tokens.dtype).repeat(batch_size, seqlen, 1)
            m_cond_tokens = torch.where(m_keep_mask_embed, m_cond_tokens, null_m_cond_embed)

            null_p_cond_hidden = self.null_p_cond_hidden.unsqueeze(0).to(p_cond_hidden.dtype)
            p_cond_hidden = torch.where(p_keep_mask_hidden, p_cond_hidden, null_p_cond_hidden)

            null_m_cond_hidden = self.null_m_cond_hidden.unsqueeze(0).to(m_cond_hidden.dtype)
            m_cond_hidden = torch.where(m_keep_mask_hidden, m_cond_hidden, null_m_cond_hidden)

        elif self.guidance_drop_mode == 'zero':
            # Reshape masks and cast to float for multiplication
            p_keep_mask_embed = p_keep_mask.view(batch_size, 1, 1).to(p_cond_tokens.dtype)
            p_keep_mask_hidden = p_keep_mask.view(batch_size, 1, 1).to(p_cond_hidden.dtype)
            m_keep_mask_embed = m_keep_mask.view(batch_size, 1, 1).to(m_cond_tokens.dtype)
            m_keep_mask_hidden = m_keep_mask.view(batch_size, 1, 1).to(m_cond_hidden.dtype)

            # Zero out the dropped conditions by multiplying with the mask (True=1, False=0)
            # Attention, this in-place operation will change the original cond_tokens and cond_hidden, cause the gradient will be lost
            # p_cond_tokens *= p_keep_mask_embed
            # m_cond_tokens *= m_keep_mask_embed
            # p_cond_hidden *= p_keep_mask_hidden
            # m_cond_hidden *= m_keep_mask_hidden
            
            # use the out-place operation to avoid the gradient loss
            p_cond_tokens = p_cond_tokens * p_keep_mask_embed
            m_cond_tokens = m_cond_tokens * m_keep_mask_embed
            p_cond_hidden = p_cond_hidden * p_keep_mask_hidden
            m_cond_hidden = m_cond_hidden * m_keep_mask_hidden

        # --- Fuse Conditions and Continue with Transformer Decoder ---
        
        cond_tokens = self.fuse_pm_tokens(torch.cat([p_cond_tokens, m_cond_tokens], dim=-1)) if not self.music_time_merged else p_cond_tokens
        
        t_hidden = self.time_mlp(timesteps)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        cond_hidden = self.fuse_pm_hidden(torch.cat([p_cond_hidden, m_cond_hidden], dim=-1)) if not self.music_time_merged else m_cond_hidden

        t = t.unsqueeze(1) + cond_hidden

        c = torch.cat((cond_tokens, t_tokens), dim=-2) if not self.music_time_merged else cond_tokens
        cond_tokens = self.norm_cond(c)

        # Corrected memory_mask creation
        memory_mask_val = torch.cat((mask, torch.zeros((batch_size, seqlen, 2), device=device, dtype=bool)), dim=-1) if not self.music_time_merged else mask
        output = self.seqTransDecoder(x, cond_tokens, t, tgt_mask=mask, memory_mask=memory_mask_val)
        output = self.final_layer(output)
        return output
