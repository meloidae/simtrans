from typing import Optional
from copy import deepcopy

from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn as nn


def get_opt_params(model: nn.Module, weight_decay: float):
    with_weight_decay = []
    without_weight_decay = []
    for name, param in list(model.named_parameters()):
        if 'bias' in name or 'Embedding' in name or 'Encoding' in name:
            without_weight_decay.append(param)
        else:
            with_weight_decay.append(param)
    opt_params = [{'params': with_weight_decay, 'weight_decay': weight_decay},
                  {'params': without_weight_decay, 'weight_decay': 0.0}]
    return opt_params


def get_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    name = cfg.optim.name
    assert name == 'adamw'
    lr_peak = cfg.optim.lr
    b0 = cfg.optim.b0
    b1 = cfg.optim.b1
    eps = cfg.optim.eps
    weight_decay = cfg.optim.weight_decay
    params = get_opt_params(model, weight_decay)
    return optim.AdamW(params, lr=lr_peak, betas=(b0, b1), eps=eps, weight_decay=weight_decay)


def get_lr_scheduler(cfg: DictConfig):
    schedule = cfg.optim.schedule
    assert schedule == 'inverse_square_root_with_warmup'
    num_warmup_steps = cfg.optim.warmup
    lr = cfg.optim.lr
    min_lr = cfg.optim.min_lr
    warmup_init_lr = cfg.optim.warmup_init_lr
    return InverseSquareRootWithWarmUpScheduler(lr=lr,
                                                warmup=num_warmup_steps,
                                                warmup_init_lr=warmup_init_lr,
                                                min_lr=min_lr)


# def get_inverse_square_root_schedule_with_warmup(optimizer: optim.Optimizer,
#                                                  num_warmup_steps: int,
#                                                  last_epoch: Optional[int] = -1):
#     def lr_lambda(current_step: int) -> float:
#         if current_step < num_warmup_steps:
#             return float(current_step / max(1.0, num_warmup_steps))
#         else:  # decay proportional to inverse square root of number of updates
#             return float(num_warmup_steps ** 0.5 * current_step ** -0.5)
# 
#     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

class InverseSquareRootWithWarmUpScheduler:
    def __init__(self, lr: float, warmup: int, warmup_init_lr: float, min_lr: float):
        self.n_step = 0
        self.lrs = torch.linspace(warmup_init_lr, lr, warmup).tolist()
        self.warmup = warmup
        self.decay_factor = lr * (self.warmup ** 0.5)
        self.min_lr = min_lr

    def __call__(self, optimizer: optim.Optimizer):
        if self.n_step < self.warmup:
            lr = self.lrs[self.n_step]
        else:
            lr = self.decay_factor / ((self.n_step + 1) ** 0.5)
        lr = max(lr, self.min_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        self.n_step += 1

    def step(self, optimizer: optim.Optimizer):
        self.__call__(optimizer)

    def state_dict(self):
        state = {'step': self.n_step}
        return {'state': state}

    def load_state_dict(self, state_dict):
        state_dict = deepcopy(state_dict)
        self.n_step = state_dict['state']['step']

