import copy
from pathlib import Path
import torch
from easydict import EasyDict
from models import FullBody_HFSQ, ReactDance
from .normalization import MotionNormalization


def read_checkpoint(path):
    path = Path(path)
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error).lower():
            raise
        return torch.load(str(path), map_location='cpu', weights_only=False)


def hfsq_model(config):
    config = EasyDict(copy.deepcopy(config))
    for part in ('up_half', 'down_half', 'tr_half'):
        for key, value in config.unified_params.items():
            if key not in config[part]:
                config[part][key] = copy.deepcopy(value)
    return FullBody_HFSQ(config, num_joints=22)


def prefixed_state(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def metadata(checkpoint, model=None):
    meta = checkpoint.get('reactdance_v2')
    if not meta or meta.get('format') != 2:
        raise ValueError('Expected a ReactDance checkpoint with model configuration and normalizers.')
    if model and meta['model'] != model:
        raise ValueError('Expected %s checkpoint, got %s' % (model, meta['model']))
    return meta


def components(model, config, normalization_state, hfsq_config=None, hfsq_state=None):
    normalizer = MotionNormalization(normalization_state)
    if model == 'HFSQ':
        return hfsq_model(config['model']), None, normalizer
    if hfsq_config is None or hfsq_state is None:
        raise ValueError('ReactDance requires the selected HFSQ model and weights.')
    hfsq = hfsq_model(hfsq_config)
    hfsq.load_state_dict(hfsq_state, strict=True)
    hfsq.requires_grad_(False).eval()
    generator = ReactDance(EasyDict(copy.deepcopy(config['model'])), normalizer,
                          decoders={'hfsq': hfsq}, use_EMA=False)
    return generator, hfsq, normalizer


def load_components(checkpoint, model):
    meta = metadata(checkpoint, model)
    state = checkpoint['state_dict']
    hfsq_config = meta.get('hfsq_config')
    hfsq_state = prefixed_state(state, 'hfsq.') if model == 'ReactDance' else None
    primary, hfsq, normalizer = components(model, meta['config'], meta['normalizers'], hfsq_config, hfsq_state)
    primary.load_state_dict(prefixed_state(state, 'model.'), strict=True)
    primary.eval()
    return primary, hfsq, normalizer, meta
