import torch
import torch.nn as nn
from models.reactdance.layers.structure_utils import zero_module
from .transformer import get_attn_block_backend
        
def get_subsequent_mask(seq_len, token_block_size=60, attn_mask_type="interval"):
    if attn_mask_type is None or attn_mask_type.lower() == "none":
        return None
    if attn_mask_type == "interval":
        # Create block indices for positions up to block_size
        idx = torch.arange(seq_len, device='cuda') // token_block_size

        # Initialize the mask with ones
        mask = torch.ones((seq_len, seq_len), device=torch.device('cuda'))

        # Set the mask to 0 where block indices are equal (same block), 1 elsewhere
        mask = (idx[:, None] == idx[None, :]).float()
        
    elif attn_mask_type == "triangle_interval":
        # Block indices for each position
        idx = torch.arange(seq_len, device=torch.device('cuda')) // token_block_size
        # Row and column indices for broadcasting
        j = torch.arange(seq_len, device=torch.device('cuda'))[None, :]  # Shape: (1, seq_len)
        i = torch.arange(seq_len, device=torch.device('cuda'))[:, None]  # Shape: (seq_len, 1)
        # Condition: same block AND j <= i
        condition = (idx[None, :] == idx[:, None]) & (j <= i)
        # Mask: 0 where condition is True, 1 where False
        mask = condition.float()
    
    elif attn_mask_type in ["full", "casual", "diag"]:
        token_block_size -= 1
        if attn_mask_type == "full":
            mask = torch.ones((seq_len, seq_len), device=torch.device('cuda'))
            mask = torch.triu(mask, diagonal=-token_block_size)
            mask = torch.tril(mask, diagonal=token_block_size)
        elif attn_mask_type == "casual":
            mask = torch.ones((seq_len, seq_len), device=torch.device('cuda'))
            mask = torch.triu(mask, diagonal=-token_block_size)
            mask = torch.tril(mask, diagonal=0)
        elif attn_mask_type == "diag":
            mask = torch.eye(seq_len, seq_len, device=torch.device('cuda'))
    else:
        raise ValueError(f"Unknown attn_mask_type: {attn_mask_type}")
    
    return (1 - mask).bool()


    return mask #.bool()
def get_sudo_triu_mask(seq_len, look_forward):
    """ For masking out the subsequent info. """
    # batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len + look_forward)).float()
    mask = 1 - torch.triu(mask, diagonal=seq_len)
    # mask = 1 - mask

    return mask #.bool()

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class LATBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, spatial_ds_rate=1):
        super().__init__()
        n_embd = int(config.n_embd * spatial_ds_rate)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, config.n_head, dropout=config.attn_pdrop, batch_first=True)
        # self.attn = MaskAttention(n_embd, config.n_head, config.attn_pdrop, config.attn_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.attn_pdrop),
        )

    def forward(self, x):
        ln_x = self.ln1(x)
        x = x + self.attn(ln_x, ln_x, ln_x, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class TemporalDownsampler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        ds_n_embd = int(config.n_embd * config.spatial_ds_rate)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.downsample = nn.Linear(config.n_cond * config.temp_ds_rate, ds_n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, cond, temporal_mask=None):
        b, t, c = cond.size()
        x = self.downsample(cond.view(b, t // self.config.temp_ds_rate, c * self.config.temp_ds_rate))
        x = self.drop(x)

        return x


class FeatureEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.music_input_attn = TemporalDownsampler(config.music_trans)
        self.leader_pos_input_attn = TemporalDownsampler(config.leader_pos_trans)
            
    def forward(self, cond, joint_num, attn_mask=None):
        music, leader_pos = cond
        music_emb = self.music_input_attn(music, attn_mask)
        leader_pos_emb = self.leader_pos_input_attn(leader_pos, attn_mask)
        return torch.cat([music_emb, leader_pos_emb], dim=-1)
        
    def music_forward(self, music, attn_mask=None):
        return self.music_input_attn(music, attn_mask)


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None, m_cond_drop_prob=0, p_cond_drop_prob=0):
        B, T, D = x.shape

        assert m_cond_drop_prob == 0 and p_cond_drop_prob == 0, "Conditional dropout not supported in CFG sampling"
        
        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(x_combined, cond=cond, timesteps=timesteps_combined, mask=mask, m_cond_drop_prob=m_cond_drop_prob, p_cond_drop_prob=p_cond_drop_prob)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond
        return cfg_out