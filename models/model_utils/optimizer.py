import numpy as np
import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

class Loss_Scaler():
    def __init__(self, use_loss_scaler = False, default_min_scale=2000, default_max_scale=10000, scale_up_steps=500, scale_up_unit=50, **kwargs):
        self.scaler = default_min_scale
        # self.register_buffer("scaler", torch.tensor(default_min_scale, dtype=torch.float32), persistent = True)
        
        self.scaler_iter = 0
        self.count_valid_steps = 0
        self.default_min_scale = default_min_scale
        self.default_max_scale = default_max_scale
        self.scale_up_steps = scale_up_steps
        self.scale_up_unit = scale_up_unit
        self.use_loss_scaler = use_loss_scaler
        
    def update_scaler(self, loss_invalid):
        # print(f'self.use_loss_scaler: {self.use_loss_scaler}')
        if not self.use_loss_scaler:
            return 1
        if loss_invalid:
            self.count_valid_steps = 0
            self.scaler = max(self.scaler >> 1, self.default_min_scale)
        else:
            self.count_valid_steps += 1
            if self.count_valid_steps == self.scale_up_steps:
                self.scaler = min(self.scaler + self.scale_up_unit, self.default_max_scale)
                self.count_valid_steps = 0
        return self.scaler

class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, **kwargs):
        """Initialize the hyperparameters.
        Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-4)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        lr = float(lr)
        
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
            the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
    
    
class Adam(torch.optim.Adam):
    r"""Implements Adam algorithm."""

    def __init__(self, 
                 params,
                 lr = 3e-5,
                 betas = (0.9, 0.999),
                 eps = 1e-6,
                 weight_decay= 0,
                 **kwargs,
                 ):

        super().__init__(params=params, lr=float(lr), betas=betas, eps=eps, weight_decay=float(weight_decay))
      
    def update_epoch(self):
        pass
    
    def step(self, closure=None):
        super().step(closure)
    
class AdamW(torch.optim.AdamW):
    r"""Implements Adam algorithm."""

    def __init__(self, 
                 params,
                 lr = 3e-5,
                 betas = (0.9, 0.999),
                 eps = 1e-6,
                 weight_decay= 0,
                 **kwargs,
                 ):

        super().__init__(params=params, lr=float(lr), betas=betas, eps=eps, weight_decay=float(weight_decay))
      
    def step(self, closure=None):
        super().step(closure)
        