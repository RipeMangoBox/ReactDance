"""Checkpoint-contained physical normalizers and the upstream identity latent rule."""
import numpy as np
import torch
from torch import nn


def dataset_normalizers(root, hfsq_config):
    paths = {
        'pos3dl': root / 'motion/pos3dl_normalizer.pt',
        'pos3df': root / 'motion/pos3df_normalizer.pt',
        'music': root / 'music/music_normalizer.pt',
    }
    result = {}
    for module, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError('Dataset normalizer missing: %s' % path)
        source = torch.load(str(path), map_location='cpu', weights_only=False)
        for kind in ('mean', 'std'):
            key = module + '-' + kind
            result[key] = source[key].float().reshape(-1).clone()
    count = hfsq_config['unified_params']['Bottleneck']['HFSQ_BottleneckBlock']['num_quantizers']
    dim = hfsq_config['dim'] * 3
    for name, size in [('quantizeds', dim), ('all_quantizeds', dim * count)]:
        result[name + '-mean'] = torch.zeros(size)
        result[name + '-std'] = torch.ones(size)
    return result


class MotionNormalization(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.modules_names = sorted({key.rsplit('-', 1)[0] for key in state})
        for module in self.modules_names:
            for kind in ('mean', 'std'):
                value = state[module + '-' + kind].detach().float().reshape(-1).clone()
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite normalizer: ' + module)
                if kind == 'std':
                    value = torch.where(abs(value) < 1e-6, value + 1e-6, value)
                self.register_buffer(module + '_' + kind, value, persistent=False)

    def export(self):
        return {module + '-' + kind: getattr(self, module + '_' + kind).detach().cpu().clone()
                for module in self.modules_names for kind in ('mean', 'std')}

    def forward(self, x, module_name, shape=None):
        if module_name is None:
            return x.clone()
        original = x.shape
        value = x.clone().reshape(shape) if shape is not None else x.clone()
        value = (value - getattr(self, module_name + '_mean')) / getattr(self, module_name + '_std')
        return value.reshape(original)

    def backward(self, x, module_name, shape=None):
        if module_name is None:
            return x.clone()
        original = x.shape
        value = x.clone().reshape(shape) if shape is not None else x.clone()
        value = value * getattr(self, module_name + '_std') + getattr(self, module_name + '_mean')
        return value.reshape(original)
