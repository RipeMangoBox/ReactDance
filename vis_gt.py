import argparse

import torch
import torch.utils

from LightningModel import LitReactDance_hfsq
from utils.config_utils import load_config, configure_device_from_config
from datasets.dd100lf_all2 import DD100lfAll


torch.set_float32_matmul_precision("medium")


def _data_loader(
    data_cfg,
    training_mode: str,
    num_workers: int,
    num_joints: int,
):
    dataset = DD100lfAll(
        training_mode,
        data_cfg.music_root,
        data_cfg.data_root,
        split=data_cfg.split,
        interval=data_cfg.interval,
        dtype=data_cfg.dtype,
        move=data_cfg.move,
        num_joints=num_joints,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )


def _dataloaders(data_cfg, num_workers: int):
    num_joints = getattr(data_cfg, "num_joints", 22)
    setattr(data_cfg.test, "batch_size", 1)
    setattr(data_cfg.test, "interval", None)
    val_dl = _data_loader(
        data_cfg.test,
        "test",
        num_workers=num_workers,
        num_joints=num_joints,
    )
    return val_dl

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ground-truth for DD100 dataset.")
    parser.add_argument(
        "--config",
        default="configs/hfsq128_512_q2.yaml",
        help="Path to hfsq configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=["vis_gt"],
        default="vis_gt",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="CUDA devices to use, e.g. '0' or '0,1'. Overrides config.devices if set.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of DataLoader workers. Overrides config.data.num_workers if set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.devices is not None:
        config.devices = args.devices
    num_workers = args.num_workers if args.num_workers is not None else getattr(config.data, "num_workers", 16)

    device = configure_device_from_config(config)

    val_dl = _dataloaders(config.data, num_workers=num_workers)
    litmodel = LitReactDance_hfsq(config, test_loader=val_dl).to(device)
    litmodel.vis_gt()

if __name__ == "__main__":
    main()
