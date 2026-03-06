import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, _LRScheduler

class StepLRWithMinLr(StepLR):
    def __init__(self, optimizer, gamma=0.1, step_size = 10, min_lr=1e-6, last_epoch=-1):
        self.min_lr = min_lr
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def step(self, epoch=None):
        super().step(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], self.min_lr) 

class LRFactor(_LRScheduler):
    def __init__(self, optimizer, decay_factor=0.99, last_epoch=-1, verbose=False, **kwargs):
        """
        初始化学习率调节器。
        :param optimizer: 优化器对象
        :param decay_factor: 每轮学习率乘以的因子，默认为 0.99
        :param last_epoch: 最后一个 epoch 的索引，默认值为 -1（重新开始计数）
        :param verbose: 是否打印学习率更新信息
        """
        
        '''
        epoch   decay_factor
        500     0.99
        1000    0.995
        2000    0.998
        '''
        self.decay_factor = decay_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        计算新的学习率。
        :return: 当前 epoch 所有参数组的学习率列表
        """
        if not self._get_lr_called_within_step:
            raise RuntimeError("get_lr should not be called directly, use step() instead.")

        if self.last_epoch == 0:
            # 初始学习率
            return [group['lr'] for group in self.optimizer.param_groups]

        # 按 decay_factor 更新学习率
        return [group['lr'] * self.decay_factor for group in self.optimizer.param_groups]
    
# new_version
class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup, cosine_epochs, stable_epochs=0, use_torch_annealing=True, default_min_lr=1e-6, eta_min=1e-6, verbose=False, **kwargs):
        self.warmup = warmup
        self.cosine_epochs = cosine_epochs + 1
        self.stable_epochs = stable_epochs  # must +1 to fit the last step()
        self.default_min_lr = float(default_min_lr)
        self.lrs = None
        self.use_torch_annealing = use_torch_annealing
        self.eta_min = float(eta_min)
        super().__init__(optimizer, verbose=verbose)
        
    def init_lrs(self):
        """Precompute all learning rates."""
        base_lr = self.base_lrs[0]  # Assume all parameter groups have the same base_lr

        # Warmup phase: Linear increase
        self.lrs = [base_lr * (epoch + 1) / self.warmup for epoch in range(self.warmup - 1)]
        
        # Cosine annealing phase
        cycle_length = self.cosine_epochs
        if self.use_torch_annealing:
            self.lrs += [
                max(self.default_min_lr, 
                    self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * epoch / self.cosine_epochs)) / 2
                ) for epoch in range(cycle_length)
            ]
        else:
            self.lrs += [
                max(self.default_min_lr, base_lr * 0.5 * (1 + np.cos(np.pi * (epoch / cycle_length))))
                for epoch in range(cycle_length)
            ]

        min_lr = max(self.default_min_lr, self.lrs[-1])
        # min_lr = self.lrs[-1]
        # Stable phase: Constant learning rate
        self.lrs += [min_lr] * self.stable_epochs
        
        done = True
    
    def get_lr(self):
        if self.lrs is None:
            self.init_lrs()  # Ensure lrs is initialized before accessing
        if self.last_epoch < len(self.lrs):
            return [self.lrs[self.last_epoch]]


class CosineWarmupRecursiveScheduler(_LRScheduler):
    '''
    use upper half of cosine curve to decay learning rate recursively
    warmup stands for finished  of initial warmup and upper half of cosine curves
    '''
    def __init__(self, optimizer, warmup, cosine_epochs, default_min_lr=1e-6, verbose=False, **kwargs):
        self.warmup = warmup
        self.cosine_epochs = cosine_epochs
        self.min_lr = default_min_lr
        super().__init__(optimizer, verbose=verbose)
    
    def get_lr(self):
        if self.last_epoch < self.warmup - 1:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup for base_lr in self.base_lrs]
        else:
            # Recursive cosine decay
            cycle_length = self.max_iters - self.warmup
            progress_in_cycle = self.last_epoch - self.warmup

            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress_in_cycle / cycle_length))
            lr = [max(self.min_lr, base_lr * cosine_decay) for base_lr in self.base_lrs]
            if progress_in_cycle * 2 >= cycle_length:
                self.warmup = self.last_epoch
                self.base_lrs = lr
            return lr
        
        
class StepLRScheduler(_LRScheduler):
    """
    适配 PyTorch 内置 _LRScheduler 的版本
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0.,
                 last_epoch: int = -1,
                 verbose: bool = False
                 ) -> None:
        
        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        
        # 调用 PyTorch 的标准父类构造函数
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        这是必须重写的核心方法，根据 self.last_epoch 计算新的学习率
        """
        # 使用内部计数器 self.last_epoch 作为时间步 t
        t = self.last_epoch

        # 预热阶段逻辑
        if t < self.warmup_t:
            # 计算从 warmup_lr_init 到 base_lr 的线性增长
            lrs = [
                self.warmup_lr_init + (base_lr - self.warmup_lr_init) * (t / self.warmup_t)
                for base_lr in self.base_lrs
            ]
        # 衰减阶段逻辑
        else:
            # 计算衰减次数
            decay_steps = (t - self.warmup_t) // self.decay_t
            lrs = [
                base_lr * (self.decay_rate ** decay_steps)
                for base_lr in self.base_lrs
            ]
            
        return lrs