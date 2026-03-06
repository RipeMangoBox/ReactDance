import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nets.encdec import Encoder, Decoder, assert_shape
from models.nets.vector_quantizer.bottleneck import Bottleneck
from models.model_utils.logger import average_metrics
from models.nets.losses import GeometricLoss
from torch.cuda.amp import autocast
from models.reactdance.layers.structure_utils import value_check

def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(x_target, x_pred, weighted=False):
    return torch.mean(torch.abs(x_pred - x_target)) if x_pred is not None else 0
    
class Part_HFSQ(nn.Module):
    def __init__(self, hps, dim=512):
        super().__init__()
        self.hps = hps
        self.dim = dim
        downs_t = hps.Encdec.downs_t
        strides_t = hps.Encdec.strides_t
        multipliers = hps.Encdec.multipliers 
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
            
        block_kwargs = dict(width=hps.Encdec.width, depth=hps.Encdec.depth, m_conv=hps.Encdec.m_conv, \
                        dilation_growth_rate=hps.Encdec.dilation_growth_rate, \
                        dilation_cycle=hps.dilation_cycle, \
                        reverse_decoder_dilation=hps.Encdec.reverse_decoder_dilation)

        self.sample_length = hps.sample_length

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = [(hps.sample_length // self.hop_lengths[0],)]

        if multipliers is None:
            multiplier = 1
        else:
            assert len(multipliers) >= 1, "Invalid number of multipliers"
            multiplier = multipliers[0]

        block_kwargs["width"] *= multiplier
        block_kwargs["depth"] *= multiplier

        # 单一 Encoder / Decoder
        self.encoder = Encoder(
            hps.pos_channel,
            dim,
            downs_t[:1],
            strides_t[:1],
            **block_kwargs,
        )
        self.decoder = Decoder(
            hps.pos_channel,
            dim,
            downs_t[:1],
            strides_t[:1],
            **block_kwargs,
        )

        # 提取bottleneck_name
        bottleneck_name = hps.Bottleneck.bottleneck_name
        # 读取对应bottleneck_name下的字段并存储在一个字典中
        kwargs = hps.Bottleneck.get(bottleneck_name, dict())
        
        self.bottleneck = Bottleneck(dim = dim, bottleneckBlock=hps.Bottleneck.bottleneck_name, **kwargs)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.commit = getattr(hps, 'commit', 0)
        # self.quantize = hps.quantize if hasattr(hps, 'quantize') else 0
        self.recon = getattr(hps, 'recon', 3)
        self.vel = getattr(hps, 'vel', 1)
        self.acc = getattr(hps, 'acc', 0)
        self.reg = getattr(hps, 'reg', 0)
        
        self.loss_type = getattr(hps, 'loss_type', "l1")
        self.n_joints = hps.pos_channel // 3

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        x = x.permute(0,2,1)
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1) if x is not None else None
        return x

    def decode(self, zs, decode_mode='indices'):
        if decode_mode == "pos":
            return zs
        '''
        output: x_pos_outs
        decode_mode:
            1. indices
            2. embs
        '''
        
        x_quantised = self.bottleneck.decode(zs, decode_mode=decode_mode)
        with autocast(enabled=False):
            x_out_pos = self.decoder(x_quantised)
        x_out_pos = self.postprocess(x_out_pos)
        
        return x_out_pos

    def encode(self, pos3d, encode_mode="all_quantizeds"):
        if encode_mode == "pos":
            return pos3d
        x_in = self.preprocess(pos3d)
        
        with autocast(enabled=False):
            x_out = self.encoder(x_in)
            
        output = self.bottleneck.encode(x_out, encode_mode=encode_mode)
        return output

    def sample(self, n_samples):
        zs = [torch.randint(0, self.dim, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):
        # Here, the decoder is transformed from 
        metrics = {}

        # Encode/Decode
        x_in = self.preprocess(x)
        
        with autocast(enabled=False):
            z = self.encoder(x_in)

        all_indices, xs_quantised, commit_loss, quantiser_metrics = self.bottleneck(z)
        x_pos_out = self.decoder(xs_quantised)
        x_out_pos = self.postprocess(x_pos_out)
        x_target_pos = x
        
        recons_loss = _loss_fn(x_target_pos, x_out_pos)
        velocity_loss = _loss_fn(x_out_pos[:, 1:] - x_out_pos[:, :-1], x_target_pos[:, 1:] - x_target_pos[:, :-1])
        acceleration_loss = _loss_fn(x_out_pos[:, 2:] + x_out_pos[:, :-2] - 2 * x_out_pos[:, 1:-1], x_target_pos[:, 2:] + x_target_pos[:, :-2] - 2 * x_target_pos[:, 1:-1])
        regularization = torch.zeros((), device=x.device)
        
        loss = recons_loss * self.recon + \
                commit_loss * self.commit + \
                velocity_loss * self.vel + \
                acceleration_loss * self.acc
            
        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            pos_loss=recons_loss,
            commit_loss=commit_loss,
            # quantize_loss=quantize_loss,
            regularization_loss=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))
            
        return x_out_pos, loss, metrics

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
        # 兼容旧 checkpoint 中 encoders/decoders/bottleneck.experts 相关命名
        keys = list(state_dict.keys())
        for key in keys:
            # encoders.0.* -> encoder.*
            if key.startswith(prefix + "encoders.0."):
                new_key = prefix + "encoder." + key[len(prefix + "encoders.0.") :]
                state_dict[new_key] = state_dict.pop(key)
            # 丢弃 encoders.1 及之后未使用的权重
            elif key.startswith(prefix + "encoders."):
                state_dict.pop(key)

        keys = list(state_dict.keys())
        for key in keys:
            # decoders.0.* -> decoder.*
            if key.startswith(prefix + "decoders.0."):
                new_key = prefix + "decoder." + key[len(prefix + "decoders.0.") :]
                state_dict[new_key] = state_dict.pop(key)
            # 丢弃 decoders.1 及之后未使用的权重
            elif key.startswith(prefix + "decoders."):
                state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,  # 放宽 strict 以忽略丢弃的多余 key
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
