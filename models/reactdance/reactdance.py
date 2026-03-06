import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.nets.nn import get_subsequent_mask
from models.gaussian.GaussianDiffusion import GaussianDiffusion
from models.reactdance.denoiser import ReactDance_Denoiser

class ReactDance(nn.Module):
    def __init__(self, config, normalizer, decoders=None, use_EMA=False):
        super().__init__()

        # 原逻辑中的属性赋值
        self.music_time_merged = getattr(config, 'music_time_merged', False)
        self.time_only = getattr(config, "time_only", False)
        self.n_actor = 3 if not self.music_time_merged else 2
        self.n_reactor = 3
        self.output_feats = np.prod(getattr(config, "pos_allq_shape", [768]))
        self.latent_emb_dim = getattr(config, 'n_embd', 512)  # 假设默认值为 512，根据实际情况调整
        self.prior_mode = getattr(config, 'prior_mode', "all_quantizeds")
        self.attn_mask_type = getattr(config, 'attn_mask_type', "interval")

        self.activation = 'gelu'
        self.diffusion_steps = getattr(config, 'diffusion_steps', 1000)
        self.beta_scheduler = getattr(config, 'beta_scheduler', 'cosine')

        self.normalizer = normalizer
        self.decoders = decoders

        unified_params = config.unified_params
        self.token_block_size = getattr(config, 'token_block_size', 60)
        self.temp_ds_rate = getattr(config, 'temp_ds_rate', 4)

        # 构造 temporal_mask, 在trianing使用 60 长度的特征块（240frame经比例为temp_ds_rate的时序压缩），滑动窗口为整个块大小。
        self.temporal_mask = get_subsequent_mask(
            seq_len=self.token_block_size,
            token_block_size=self.token_block_size,
            attn_mask_type=self.attn_mask_type
        )

        # unified params 合并逻辑保持不变
        block_keys = ['gpt', 'music_trans', 'leader_pos_trans']
        for block_key in block_keys:
            for u_k, u_v in unified_params.items():
                if hasattr(config, block_key) and u_k not in config[block_key].keys():
                    config[block_key][u_k] = u_v

        gpt_config = config.gpt
        self.geo_loss_checklist = config.geo_loss_checklist

        self.m_cond_drop_prob = getattr(config, 'm_cond_drop_prob', 0.1)
        self.p_cond_drop_prob = getattr(config, 'p_cond_drop_prob', 0.1)

        self.net = ReactDance_Denoiser(
            input_feats=gpt_config.n_cond,
            latent_dim=gpt_config.n_embd,
            ff_size=gpt_config.ff_size,
            num_layers=gpt_config.n_layer,
            num_heads=gpt_config.n_head,
            dropout=gpt_config.dropout,
            activation=getattr(F, self.activation),
            n_actor=self.n_actor,
            n_reactor=self.n_reactor,
            music_time_merged=self.music_time_merged,
            time_only=self.time_only,
            config=config,
        )

        self.m_guidance_weight = 1
        self.pm_guidance_weight = 1

        self.diffusion = GaussianDiffusion(
            model=self.net,
            normalizer=self.normalizer,
            decoders=self.decoders,
            n_timestep=self.diffusion_steps,
            geo_loss_checklist=self.geo_loss_checklist,
            config=config,
            use_EMA=use_EMA,
        )

    def forward(self, x_start, cond_tuple, motion_target=None, **kwargs):
        '''
        x_start: (x_start_up, x_start_down, x_start_transl)
        attention, change ([b, d, n]) to ([b, n, d])
        '''
        preds, losses, metrics = self.diffusion.forward(
            x_start=x_start,
            cond_tuple=cond_tuple,
            motion_target=motion_target,
            train_mode="train" if self.training else "val",
            model_kwargs={
                "mask":self.temporal_mask,
                "m_cond_drop_prob": self.m_cond_drop_prob if self.training else 0,
                "p_cond_drop_prob": self.p_cond_drop_prob if self.training else 0,
                "token_block_size": self.token_block_size,
            },
        )
        
        # preds have been denoimalized in self.diffusion        
        return preds, losses, metrics
    
    def sample(self, use_EMA=False, cond_tuple=None, **kwargs):
        B = cond_tuple[0].shape[0]
        seq_len = cond_tuple[0].size(1)//self.temp_ds_rate
        temporal_mask = get_subsequent_mask(seq_len=seq_len, token_block_size=self.token_block_size, attn_mask_type=self.attn_mask_type)
        
        pred_pose = self.diffusion.ddim_sample_loop(
        # pred_pose = self.diffusion.p_sample_loop(
            (B, seq_len, self.output_feats),
            cond_tuple=cond_tuple,
            use_EMA=use_EMA,
            progress=False,
            model_kwargs={
                "mask":temporal_mask,
                "token_block_size": self.token_block_size,
                # "pm_guidance_weight": self.pm_guidance_weight,
            },)
        
        return pred_pose

    def sample_cfg(self, use_EMA=False, cond_tuple=None, pm_guidance_weight=5,body_parts=['up', 'down', 'tr'], **kwargs):
        B = cond_tuple[0].shape[0]
        seq_len = cond_tuple[0].size(1)//self.temp_ds_rate
        temporal_mask = get_subsequent_mask(seq_len=seq_len, token_block_size=self.token_block_size, attn_mask_type=self.attn_mask_type)
        
        pred_pose = self.diffusion.ddim_sample_loop(
        # pred_pose = self.diffusion.p_sample_loop(
            (B, seq_len, self.output_feats),
            cond_tuple=cond_tuple,
            use_EMA=use_EMA,
            progress=False,
            model_kwargs={
                "mask":temporal_mask,
                "token_block_size": self.token_block_size,
                "pm_guidance_weight": pm_guidance_weight,
                "body_parts": body_parts,
            },)
        
        return pred_pose