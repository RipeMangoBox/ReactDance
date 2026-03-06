import torch
import torch.utils
import lightning.pytorch as pl
import argparse

from LightningModel import LitReactDance_hfsq, CopyConfigCallback
from utils.config_utils import load_config, configure_device_from_config
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

def dataloaders(data_cfg, num_workers=8):
    shuffle = getattr(data_cfg.train, "shuffle", False)
    num_joints = getattr(data_cfg, 'num_joints', 22)
    rotate_prob = getattr(data_cfg.train, 'rotate_prob', 0)
    mirror_prob = getattr(data_cfg.train, 'mirror_prob', 0)
    swap_prob = getattr(data_cfg.train, 'swap_prob', 0)
    
    train_dl = data_loader(data_cfg.train, 'train', num_workers, shuffle=shuffle, rotate_prob=rotate_prob, mirror_prob=mirror_prob, swap_prob=swap_prob, num_joints=num_joints)    
    val_dl = data_loader(data_cfg.test, 'test', num_workers, shuffle=False, num_joints=num_joints)
    
    return train_dl, val_dl

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance')
    parser.add_argument(
        '--config',
        default='configs/hfsq128_512_q2.yaml',  # results/training/HFSQ/lightning_logs/hfsq128_512_q2/hfsq128_512_q2.yaml
        help='Path to training configuration file.',
    )
    parser.add_argument(
        '--mode',
        choices=[
            'train',
            'single_sample',    # reconstruct the reactor motion only
            'duet_sample',      # reconstruct the reactor motion with combined leader motion
        ],
        default='train',
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


def train_mode(config, config_path, train_dl, val_dl):
    if config.Train["resume_from_checkpoint"] is not None:
        ckpt = config.Train["resume_from_checkpoint"]
        print(f"resuming from checkpoint: {ckpt}")
        
        litmodel = LitReactDance_hfsq.load_from_checkpoint(ckpt, config=config, test_loader=val_dl)
        print("checkpoint state loaded!")
    else:
        litmodel = LitReactDance_hfsq(config, test_loader=val_dl)
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
    copy_config_callback = CopyConfigCallback(config_path)
    
    scheduler_config = config.Train.scheduler
    epochs = getattr(scheduler_config, "total_epochs", 301)
    
    trainer = pl.Trainer(
        strategy= "ddp" if "ddp" in config_path else None,
        callbacks=[epoch_checkpoint, copy_config_callback],
        max_epochs=epochs,
        **(trainer_params["Trainer"]),
    )

    trainer.fit(model=litmodel, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt)

def single_sample_mode(config, litmodel):
    litmodel.single_synthesis(no_video = False)
    
def duet_sample_mode(config, litmodel):
    litmodel.duet_synthesis(no_video = False)

def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.devices is not None:
        config.devices = args.devices
    num_workers = args.num_workers if args.num_workers is not None else getattr(config.data, "num_workers", 8)
    if args.batch_size is not None and hasattr(config.data, "train"):
        setattr(config.data.train, "batch_size", args.batch_size)

    print(config)
    pl.seed_everything(325514)

    train_dl, val_dl = dataloaders(config.data, num_workers=num_workers)
    device = configure_device_from_config(config)

    def _load_litmodel_from_checkpoint() -> LitReactDance_hfsq:
        ckpt = getattr(config.data.test, "checkpoint", None)
        if ckpt is None:
            raise ValueError("config.data.test.checkpoint must be set for this mode.")
        print(f"Loading LitReactDance_hfsq from checkpoint: {ckpt}")
        return LitReactDance_hfsq.load_from_checkpoint(ckpt, config=config, test_loader=val_dl).to(device)

    mode_switch = {
        'train': lambda: train_mode(config, args.config, train_dl, val_dl),
        'single_sample': lambda: single_sample_mode(config, _load_litmodel_from_checkpoint()),
        'duet_sample': lambda: duet_sample_mode(config, _load_litmodel_from_checkpoint()),
    }

    if args.mode not in mode_switch:
        raise ValueError(f"Unknown mode: {args.mode}")

    mode_switch[args.mode]()


if __name__ == '__main__':
    main()
