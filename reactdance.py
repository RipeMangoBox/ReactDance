import torch.utils
import torch
import lightning.pytorch as pl
import os
import argparse
from easydict import EasyDict
import sys

from LightningModel import (
    LitReactDance_reactdance,
    LitReactDance_hfsq,
    CopyConfigCallback,
    ProfilerCallback,
)
from utils.config_utils import load_config, configure_device_from_config
from utils.checkpoint_loading import cfg_reader
from datasets.dd100lf_all2 import DD100lfAll

torch.set_float32_matmul_precision('medium')

def data_loader(data_cfg, training_mode, num_workers=16, shuffle=False, num_joints=22, rotate_prob=0, mirror_prob=0, swap_prob=0):
    dataset = DD100lfAll(
        training_mode,
        data_cfg.music_root, 
        data_cfg.data_root, 
        split=data_cfg.split, 
        interval=data_cfg.interval, 
        dtype=data_cfg.dtype, 
        move=data_cfg.move, 
        num_joints=num_joints,
        rotate_prob=rotate_prob,
        mirror_prob=mirror_prob,
        swap_prob=swap_prob,
        )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True,
    )

def dataloaders(data_cfg, num_workers=16):
    shuffle = getattr(data_cfg.train, "shuffle", False)
    num_joints = getattr(data_cfg, 'num_joints', 22)
    rotate_prob = getattr(data_cfg.train, 'rotate_prob', 0)
    mirror_prob = getattr(data_cfg.train, 'mirror_prob', 0)
    swap_prob = getattr(data_cfg.train, 'swap_prob', 0)
    
    # get partationed train/val data
    train_dl = data_loader(data_cfg.train, 'train', num_workers, shuffle=shuffle, rotate_prob=rotate_prob, mirror_prob=mirror_prob, swap_prob=swap_prob, num_joints=num_joints)    
    val_dl = data_loader(data_cfg.test, 'val', num_workers, shuffle=False, num_joints=num_joints)
    
    # get full length test data
    setattr(data_cfg.test, 'interval', None)
    setattr(data_cfg.test, 'batch_size', 1)
    test_dl = data_loader(data_cfg.test, 'test', num_workers, shuffle=False, num_joints=num_joints)
    return train_dl, val_dl, test_dl


def _build_hfsq_train_loader(hfsq_cfg, num_workers: int):
    """
    Build the stage-1 (HFSQ) train loader for normalizer calculation.
    Mirrors `hfsq.save_mean_std` behavior: no shuffle, no aug.
    """
    data_cfg = hfsq_cfg.data
    dataset = DD100lfAll(
        "train",
        data_cfg.train.music_root,
        data_cfg.train.data_root,
        split=data_cfg.train.split,
        interval=data_cfg.train.interval,
        dtype=data_cfg.train.dtype,
        move=data_cfg.train.move,
        num_joints=data_cfg.num_joints,
        rotate_prob=0,
        mirror_prob=0,
        swap_prob=0,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.train.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

def get_hfsq_normalizer(gen_cfg: EasyDict, device: torch.device, num_workers: int):
    """
    Stage-2 bootstrap:
    - Load stage-1 config (gen_cfg.HFSQ_config)
    - Derive HFSQ checkpoint path from stage-1 config.data.test.checkpoint
    - If normalizer file is missing, auto-calc & save it with HFSQ model
    """
    hfsq_cfg_path = gen_cfg.HFSQ_config
    hfsq_cfg = cfg_reader(hfsq_cfg_path)
    hfsq_ckpt = getattr(hfsq_cfg.data.test, "checkpoint", None)
    if hfsq_ckpt is None:
        raise ValueError(
            "Stage-1 config is missing `data.test.checkpoint`; please set it to a HFSQ ckpt path."
        )

    hfsq_normalizer_dir = hfsq_ckpt.replace("checkpoints", "normalizers")[:-5]  # remove .ckpt
    hfsq_normalizer_file = getattr(
        getattr(gen_cfg.structure_generate, gen_cfg.structure_generate.generator_name, EasyDict()),
        "hfsq_normalizer_file",
        "normalizer.pt",
    )
    normalizer_path = os.path.join(hfsq_normalizer_dir, hfsq_normalizer_file)

    if os.path.exists(normalizer_path):
        print(f"[Stage2] HFSQ normalizer exists: {normalizer_path}")
        return

    print(f"[Stage2] HFSQ normalizer missing, auto-calculating: {normalizer_path}")

    train_dl = _build_hfsq_train_loader(hfsq_cfg, num_workers=num_workers)

    # Load HFSQ lightning module to reuse its `save_mean_std` implementation
    ckpt_obj = torch.load(hfsq_ckpt, map_location=device)
    lit_hfsq = LitReactDance_hfsq(hfsq_cfg, test_loader=None).to(device)
    lit_hfsq.load_state_dict(ckpt_obj["state_dict"], strict=False)

    lit_hfsq.save_mean_std(
        train_loader=train_dl,
        device=str(device),
        modes=["quantizeds", "all_quantizeds"],
        offline=True,
    )

    if not os.path.exists(normalizer_path):
        raise RuntimeError(
            f"Normalizer auto-calc finished but file not found at: {normalizer_path}"
        )
    print(f"[Stage2] HFSQ normalizer saved: {normalizer_path}")

def get_train_inference_dataloader(data_cfg, num_workers=16):
    setattr(data_cfg.train, 'interval', None)
    setattr(data_cfg.train, 'batch_size', 1)
    train_dl = data_loader(data_cfg.train, 'train', num_workers)
    return train_dl

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance'
    )
    parser.add_argument(
        '--config',
        default='configs/reactdance.yaml', # 'results/training/ReactDance/lightning_logs/reactdance/reactdance.yaml',
        help='Path to generator training configuration file.',
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'sample'],
        default='sample',
        help='Running mode.',
    )
    parser.add_argument(
        '--devices',
        type=str,
        default=None,
        help="CUDA devices to use, e.g. '0' or '0,1'. Overrides config.devices if set.",
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of DataLoader workers. Overrides config.data.num_workers if set.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override training batch size defined in config.data.train.batch_size.',
    )
    return parser.parse_args()


def train_mode(config, config_path, train_dl, val_dl, test_dl):
    if config.Train["resume_from_checkpoint"] is not None:
        ckpt = config.Train["resume_from_checkpoint"]
        print(f"resuming from checkpoint: {ckpt}")
        
        litmodel = LitReactDance_reactdance.load_from_checkpoint(ckpt, config=config, val_dl=val_dl, test_dl=test_dl)
        print("checkpoint state loaded!")
    else:
        litmodel = LitReactDance_reactdance(config, val_dl=val_dl, test_dl=test_dl)
        ckpt = None

    trainer_params = vars(config).copy()

    # 仅按照 epoch 间隔保存 checkpoint，不再按指标排序
    checkpoint_interval = getattr(config.Train, "checkpoint_interval", 25)
    epoch_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=checkpoint_interval,
        save_last=False,
        filename="epoch{epoch:03d}",
        verbose=True,
    )
    
    # 实例化 CopyConfigCallback 并传入配置文件路径
    copy_config_callback = CopyConfigCallback([config_path, config.HFSQ_config])
    scheduler_config = config.Train.scheduler
    epochs = getattr(scheduler_config, "total_epochs", 501)
    
    trainer = pl.Trainer(
        strategy= "ddp" if "ddp" in config_path else None,
        callbacks=[epoch_checkpoint, copy_config_callback],
        max_epochs=epochs,
        **(trainer_params["Trainer"]),
        # num_sanity_val_steps=0, # skip sanity check
    )

    trainer.fit(model=litmodel, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt)

def sample_mode(config, litmodel, test_dl):
    litmodel.synthesis(
        test_loader=test_dl,
        no_video=False,
        sample_after_train=True,
        pm_guidance_weight=1.2,
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    print(config)

    # CLI overrides
    if args.devices is not None:
        config.devices = args.devices
    num_workers = args.num_workers if args.num_workers is not None else getattr(config.data, "num_workers", 16)
    if args.batch_size is not None and hasattr(config.data, "train"):
        setattr(config.data.train, "batch_size", args.batch_size)

    random_seed = config.RANDOM_SEED if hasattr(config, "RANDOM_SEED") and config.RANDOM_SEED != 'None' else None
    pl.seed_everything(random_seed)

    train_dl, val_dl, test_dl = dataloaders(config.data, num_workers=num_workers)

    device = configure_device_from_config(config)

    # Stage-2 bootstrap: auto-check & auto-generate HFSQ normalizer if missing.
    # This removes the manual `hfsq.py --mode calc_normalizer` step.
    get_hfsq_normalizer(config, device=device, num_workers=num_workers)

    def _load_litmodel_from_checkpoint() -> LitReactDance_reactdance:
        ckpt = getattr(config.data.test, "checkpoint", None)
        if ckpt is None:
            raise ValueError("config.data.test.checkpoint must be set for 'sample' mode.")
        print(f"mode:[{args.mode}], ckpt load from checkpoint: {ckpt}")
        litmodel = LitReactDance_reactdance.load_from_checkpoint(
            ckpt, config=config, val_dl=val_dl, test_dl=test_dl, strict=False
        ).to(device)
        print("checkpoint state loaded!")
        return litmodel

    mode_switch = {
        'train': lambda: train_mode(config, args.config, train_dl, val_dl, test_dl),
        'sample': lambda: sample_mode(config, _load_litmodel_from_checkpoint(), test_dl),
    }

    if args.mode not in mode_switch:
        raise ValueError(f"Unknown mode: {args.mode}")

    mode_switch[args.mode]()


if __name__ == '__main__':
    main()
