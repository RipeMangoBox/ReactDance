import torch as t
import torch.nn as nn
from .resnet import Resnet, Resnet1D
from models.model_utils.torch_utils import assert_shape
import torch
import torch.nn as nn

class DenoiseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=max(1, channels // 8), num_channels=channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=max(1, channels // 8), num_channels=channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class NoiseInjection(nn.Module):
    def __init__(self, noise_level=0.01):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x  # 在推理时不注入噪声

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False,noise_level=0.0, final_conv=False):
        super().__init__()
        self.noise_inject = NoiseInjection(noise_level)  # 新增：噪声注入
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.noise_inject(x)  # 注入噪声增强鲁棒
        return self.model(x)

class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        first_down_t = downs_t[0]
        first_stride_t = strides_t[0]
        self.block = EncoderConvBlock(
            input_emb_width,
            output_emb_width,
            first_down_t,
            first_stride_t,
            final_conv=True,
            **block_kwargs_copy,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # 兼容旧 checkpoint 中的 level_blocks.0.* 命名
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(prefix + "level_blocks.0."):
                new_key = prefix + "block." + key[len(prefix + "level_blocks.0.") :]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith(prefix + "level_blocks."):
                # 丢弃 level_blocks.1 及之后未使用的权重
                state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,  # 放宽 strict 以忽略多余 key
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.downs_t = downs_t
        self.strides_t = strides_t

        first_down_t = downs_t[0]
        first_stride_t = strides_t[0]
        self.block = DecoderConvBock(
            output_emb_width,
            output_emb_width,
            first_down_t,
            first_stride_t,
            **block_kwargs,
        )

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # 兼容旧 checkpoint 中的 level_blocks.0.* 命名
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(prefix + "level_blocks.0."):
                new_key = prefix + "block." + key[len(prefix + "level_blocks.0.") :]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith(prefix + "level_blocks."):
                # 丢弃 level_blocks.1 及之后未使用的权重
                state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,  # 放宽 strict 以忽略多余 key
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        x = self.block(x)
        x = self.out(x)
        return x