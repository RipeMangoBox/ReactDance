import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.dd100lf_all2 import DD100lfAll
from .paths import project_path


class MotionView(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        result = {}
        length = min(len(item['pos3dl']), len(item['pos3df']), len(item['music']))
        length = length // 4 * 4
        if length < 4:
            raise ValueError('Motion too short: ' + item['fname'])
        for key, value in item.items():
            result[key] = value[:length].copy() if isinstance(value, np.ndarray) else value
        result['sample_index'] = index
        return result


def dataset(config, kind):
    root = project_path(config['data']['root'])
    options = config['data'][kind]
    source = DD100lfAll('train' if kind == 'train' else 'test',
        str(root / 'music'), str(root / 'motion'), split=options['split'],
        interval=options.get('interval') if kind != 'test' else None,
        move=options.get('move', 4), dtype=['pos3d'], num_joints=22,
        rotate_prob=0, mirror_prob=0, swap_prob=0)
    if not len(source):
        raise ValueError('No valid samples under %s' % root)
    return MotionView(source)


def test_references(config, records):
    source = dataset(config, 'test')
    by_name = {name: index for index, name in enumerate(source.source.names)}
    if len(records) != len(by_name) or {record['sample'] for record in records} != by_name.keys():
        raise ValueError('The predictions must cover the full test set.')
    return source, by_name


def loader(config, kind, world=1, rank=0):
    source = dataset(config, kind)
    training = kind == 'train'
    batch = config['data'][kind].get('batch_size', 1) if kind != 'test' else 1
    workers = config['data'].get('num_workers', 4)
    sampler = None
    if training and world > 1:
        sampler = torch.utils.data.DistributedSampler(source, num_replicas=world, rank=rank,
            shuffle=True, seed=config['seed'], drop_last=True)
    return DataLoader(source, batch_size=batch, shuffle=training and sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=False, drop_last=training,
        persistent_workers=workers > 0)
