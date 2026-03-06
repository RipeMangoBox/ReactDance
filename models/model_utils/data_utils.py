import os
import numpy as np
import torch
import torch.nn as nn
import math

def sum_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

def count_parameters(module: nn.Module) -> float:
    num_params = sum(p.numel() for p in module.parameters())
    return round(num_params / 1e6, 3)

def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


class MotionNormalizerTorch:
    def __init__(self,
                 calculator="Z_SCORE",
                 hfsq_normalizer_file=None,
                 music_normalizer_file='data_lazy/music/music_normalizer.pt',
                 normalizer_device="cuda"):
        
        self.calculator = calculator  # Store calculator type
        self.mean_type = "-mean" 
        self.std_type = "-std"
        
        self.hfsq_normalizer_file = hfsq_normalizer_file
        self.music_normalizer_file = music_normalizer_file
        self.pos3df_normalizer_file = 'data_lazy/motion/pos3df_normalizer.pt'
        self.pos3dl_normalizer_file = 'data_lazy/motion/pos3dl_normalizer.pt'
        self.device = normalizer_device
        self.init()
        
    def _normalizer_loading(self, normalizer_file):
        if os.path.exists(normalizer_file):
            normalizer_dict = torch.load(normalizer_file, map_location=self.device)
        else:
            normalizer_dict = {}
        return normalizer_dict

    def init(self):
        mean_normalizers = {}
        std_normalizers = {}

        # Load base normalizers for pos3df from a single dict file.
        if self.pos3df_normalizer_file:
            pos3df_norm = self._normalizer_loading(self.pos3df_normalizer_file)
            mean_key = f'pos3df{self.mean_type}'
            std_key = f'pos3df{self.std_type}'
            if mean_key not in pos3df_norm or std_key not in pos3df_norm:
                raise KeyError(
                    f"Expected keys '{mean_key}' and '{std_key}' in {self.pos3df_normalizer_file}, but got keys: {list(pos3df_norm.keys())}"
                )
            mean_normalizers['pos3df'] = pos3df_norm[mean_key]
            std_normalizers['pos3df'] = pos3df_norm[std_key]
        
        # Load normalizers for reactor_hfsq
        hfsq_normalizer_dict = self._normalizer_loading(self.hfsq_normalizer_file)
        for k, v in hfsq_normalizer_dict.items():
            module_name = k.split('-')[0]
            if k.endswith(self.mean_type):
                mean_normalizers[module_name] = v.view(-1)
                # set to normalized zero
                mean_normalizers[module_name] = torch.zeros_like(mean_normalizers[module_name])
            elif k.endswith(self.std_type):
                std_normalizers[module_name] = v.view(-1)
                # set to normalized one
                std_normalizers[module_name] = torch.ones_like(std_normalizers[module_name])

        # load normalizers for leader
        pos3dl_normalizer_dict = self._normalizer_loading(self.pos3dl_normalizer_file)
        for k, v in pos3dl_normalizer_dict.items():
            module_name = k.split('-')[0]
            if k.endswith("mean"):
                mean_normalizers[module_name] = v.view(-1)
            elif k.endswith("std"):
                std_normalizers[module_name] = v.view(-1)

        pos3dl_mean = mean_normalizers["pos3dl"]
        pos3dl_std = std_normalizers["pos3dl"]
        mean_normalizers["pos3dl"] = pos3dl_mean
        std_normalizers["pos3dl"] = pos3dl_std

        # Load normalizers for music
        if self.music_normalizer_file:
            music_normalizer_dict = self._normalizer_loading(self.music_normalizer_file)
            for k, v in music_normalizer_dict.items():
                module_name = k.split('-')[0]
                if k.endswith(self.mean_type):
                    mean_normalizers[module_name] = v.view(-1)
                elif k.endswith(self.std_type):
                    std_normalizers[module_name] = v.view(-1)
        
        # Assign normalizers based on calculator type
        self.means = {k: v.to(torch.float32).to(self.device) for k, v in mean_normalizers.items()}
        self.stds = {k: v.to(torch.float32).to(self.device) for k, v in std_normalizers.items()}
        for name, vals in self.stds.items():
            self.stds[name] = torch.where(abs(vals) < 1e-6, vals + 1e-6, vals)
        
        print('normalizer loaded!')
       
    def forward(self, x, module_name, shape=None):
        x = x.clone()
        if module_name is None:
            return x
        
        if shape is not None:
            x_shape = x.shape
            x = x.view(shape)
            x = (x - self.means[module_name]) / self.stds[module_name]
            x = x.view(x_shape)
        else:
            x = (x - self.means[module_name]) / self.stds[module_name]
        return x

    def backward(self, x, module_name, shape=None):
        x = x.clone()
        if module_name is None:
            return x
        
        if shape is not None:
            x_shape = x.shape
            x = x.view(shape)
            x = x * self.stds[module_name] + self.means[module_name]
            x = x.view(x_shape)
        else:
            x = x * self.stds[module_name] + self.means[module_name]
        return x
    