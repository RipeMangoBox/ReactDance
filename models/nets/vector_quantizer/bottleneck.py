import torch
import torch.nn as nn

from models.nets.vector_quantizer.HFSQ import HFSQ
from models.nets.vector_quantizer.residual_vq import ResidualVQ

# def exists(val):
#     '''for debug'''
#     assert val is not None
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d
    
class HFSQ_BottleneckBlock(nn.Module):    
    def __init__(self, *, dim, levels, groups, num_quantizers, use_group_norm, group_norm_type, base_mask_rate=0.0, **kwargs):
        super().__init__()
        
        self.hfsq = HFSQ(
            dim = dim,
            levels = levels,
            groups = groups,
            num_quantizers = num_quantizers,
            channel_last = False,
            group_norm_type = group_norm_type,
            use_group_norm = use_group_norm,
            base_mask_rate = base_mask_rate,
            
            **kwargs['RFSQ']
        )
        
    def encode(self, z, encode_mode="all_quantizeds"):
        output = self.hfsq.encode(z, encode_mode=encode_mode)
        return output
    
    def decode(self, z, decode_mode='indices'):
        embs = self.hfsq.decode(z, decode_mode)
        return embs
    
    def forward(self, x): 
        quantized, all_indices, commit_loss, all_metrics = self.hfsq(x)
        
        flattened_metrics = {}
        for k, v in all_metrics.items():
            for group in range(v.size(0)):            # group level
                for quantizer in range(v.size(1)):    # quantizer level
                    grk = f'{k}_group{group}_quantizer{quantizer}'
                    flattened_metrics[grk] = v[group][quantizer]
                    
        return all_indices, quantized, commit_loss, flattened_metrics
    
class RVQ_BottleneckBlock(nn.Module):    
    def __init__(self, sample_codebook_temp=0.5, **kwargs):
        super().__init__()
        
        self.sample_codebook_temp = sample_codebook_temp
        self.model = ResidualVQ(
            **kwargs,
        )
        
    def encode(self, z, encode_mode="all_quantizeds"):
        output = self.model.encode(z, encode_mode=encode_mode)
        return output
    
    def decode(self, z, decode_mode='all_quantizeds'):
        output = self.model.decode(z, decode_mode=decode_mode)
        return output
    
    def forward(self, x): 
        quantized, all_indices, commit_loss, all_metrics = self.model(x, sample_codebook_temp=self.sample_codebook_temp)
        return all_indices, quantized, commit_loss, all_metrics

bottleneck_switch = {
    'HFSQ_BottleneckBlock': HFSQ_BottleneckBlock,
    'RVQ_BottleneckBlock': RVQ_BottleneckBlock,
}
    
class Bottleneck(nn.Module):
    def __init__(self, bottleneckBlock="FSQ_BottleneckBlock", **kwargs):
        super().__init__()

        if isinstance(bottleneckBlock, str):
            bottleneckBlock_class = bottleneck_switch[bottleneckBlock]
        elif not issubclass(bottleneckBlock_class, nn.Module):
            raise ValueError(
                f"bottleneckBlock must be a subclass of nn.Module or a string, got {bottleneckBlock_class}"
            )

        self.block = bottleneckBlock_class(**kwargs)
        self.use_duo = bottleneckBlock == "Duolando_BottleneckBlock"

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
        # 兼容旧 checkpoint 中的 num_experts 与 experts.0.* 命名
        keys = list(state_dict.keys())
        for key in keys:
            if key == prefix + "num_experts":
                # 不再需要 num_experts，直接丢弃
                state_dict.pop(key)
            elif key.startswith(prefix + "experts.0."):
                new_key = prefix + "block." + key[len(prefix + "experts.0.") :]
                state_dict[new_key] = state_dict.pop(key)
            elif key.startswith(prefix + "experts."):
                # 丢弃 experts.1 及之后未使用的权重
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

    def encode(self, xs, encode_mode="all_quantizeds"):
        output = self.block.encode(xs, encode_mode=encode_mode)
        return output

    def decode(self, zs, decode_mode='indices'):
        xs_quantised = self.block.decode(zs, decode_mode)
        return xs_quantised

    def forward(self, xs): # xs.shape (b, d, interval//4), '4' stands for twice downsamples
        metrics = []
        z, x_quantised, commit_loss, metric = self.block(xs)
        
        if not self.training:
            # Be extra paranoid and make sure the encoder weights can't
            # change from straight-through estimator
            x_quantised = x_quantised.detach()
        
        if self.training:
            metrics.append(metric)
                
        return z, x_quantised, commit_loss, metrics
