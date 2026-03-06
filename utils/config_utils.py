import os
from typing import Any, Dict

import torch
import yaml
from easydict import EasyDict


def load_config(path: str) -> EasyDict:
    """
    Load a YAML configuration file and wrap it in EasyDict.
    
    This helper centralizes basic validation so entry scripts and training code
    share the same behavior.
    """
    with open(path, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(raw_cfg)

    # Basic required sections for this project
    if not hasattr(config, "data"):
        raise ValueError("Config missing required 'data' section.")
    if not hasattr(config, "Train"):
        raise ValueError("Config missing required 'Train' section.")
    if not hasattr(config, "structure") and not hasattr(config, "structure_generate"):
        raise ValueError("Config must contain 'structure' or 'structure_generate' section.")

    return config


def configure_device_from_config(config: EasyDict) -> torch.device:
    """
    Configure CUDA_VISIBLE_DEVICES from config.devices (if provided) and return the torch.device.
    """
    devices = getattr(config, "devices", None)
    if devices is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device} (config.devices not set)")
        return device

    if isinstance(devices, (list, tuple)):
        devices_str = ",".join(map(str, devices))
    else:
        devices_str = str(devices)

    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Using device {device} with CUDA_VISIBLE_DEVICES={devices_str}")
    return device


def build_experiment_dirs(exp_name: str, checkpoint_path: str, base_dir: str = "./results/generated"):
    """
    Build standard experiment directories (expdir, recon, gt, samples) based on
    experiment name and checkpoint path.
    """
    experiment_dir = os.path.join(base_dir, exp_name)
    param_setting = checkpoint_path.split('/')[-3]
    checkpoint_setting = checkpoint_path.split('/')[-1]

    expdir = os.path.join(experiment_dir, param_setting, checkpoint_setting)
    sampledir = os.path.join(expdir, "samples")

    for d in (expdir, sampledir):
        os.makedirs(d, exist_ok=True)

    return expdir, sampledir

