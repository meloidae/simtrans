from typing import Optional, Tuple

from omegaconf import DictConfig
import torch
import torch.optim as optim
import torch.nn as nn

from .model import ReinforceModel
from ..optim import InverseSquareRootWithWarmUpScheduler

def get_opt_params(model: ReinforceModel, weight_decay: float):
    with_weight_decay_nmt = []
    without_weight_decay_nmt = []
    with_weight_decay_bse = []
    without_weight_decay_bse = []

    for name, mod in model.named_children():
        if name == 'bse':
            with_decay = with_weight_decay_bse
            without_decay = without_weight_decay_bse
        else:
            with_decay = with_weight_decay_nmt
            without_decay = without_weight_decay_nmt
        for n, param in mod.named_parameters():
            if 'bias' in n or 'Embedding' in n or 'Encoding' in n:
                without_decay.append(param)
            else:
                with_decay.append(param)

    params_nmt =[{'params': with_weight_decay_nmt, 'weight_decay': weight_decay},
                 {'params': without_weight_decay_nmt, 'weight_decay': 0.0}]
    params_bse =[{'params': with_weight_decay_bse, 'weight_decay': weight_decay},
                 {'params': without_weight_decay_bse, 'weight_decay': 0.0}]

    return params_nmt, params_bse


def get_optimizer(cfg: DictConfig,
                  model: ReinforceModel
                  ) -> Tuple[optim.Optimizer, Optional[optim.Optimizer]]:
    weight_decay = cfg.optim.weight_decay
    
    params_nmt, params_bse = get_opt_params(model, weight_decay=weight_decay)

    # Optimizer for NMT
    if cfg.optim.name == 'sgd':
        opt_nmt = optim.SGD(params_nmt,
                            lr=cfg.optim.lr_nmt,
                            weight_decay=weight_decay,
                            momentum=cfg.optim.momentum)
    else:  # adamw
        assert cfg.optim.name == 'adamw', f'Unimplemented optim name {cfg.optim.name}'
        opt_nmt = optim.AdamW(params_nmt,
                              lr=cfg.optim.lr_nmt,
                              betas=(cfg.optim.b0, cfg.optim.b1),
                              eps=cfg.optim.eps,
                              weight_decay=weight_decay)

    if cfg.model.baseline != 'self':
        # Optimizer for baseline estimator
        # TODO: implement SGD for bse as well?
        opt_bse = optim.AdamW(params_bse,
                              lr=cfg.optim.lr_bse,
                              weight_decay=weight_decay)
    else:
        opt_bse = None

    return opt_nmt, opt_bse

def get_lr_scheduler(cfg: DictConfig,
                     opt_nmt: optim.Optimizer,
                     opt_bse: Optional[optim.Optimizer]):
    if cfg.optim.schedule == 'reduce_on_plateau':
        nmt_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_nmt,
                                                             factor=cfg.optim.lr_decay,
                                                             patience=cfg.optim.patience)
    else:
        assert cfg.optim.schedule == 'inverse_square_root_with_warmup', f'Unimplemented lr scheduler {cfg.optim.schedule}'
        nmt_scheduler = InverseSquareRootWithWarmUpScheduler(lr=cfg.optim.lr_nmt,
                                                             warmup=cfg.optim.warmup,
                                                             warmup_init_lr=cfg.optim.warmup_init_lr,
                                                             min_lr=cfg.optim.min_lr)
    if opt_bse is not None:
        if cfg.optim.schedule == 'reduce_on_plateau':
            bse_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_bse,
                                                                 factor=cfg.optim.lr_decay,
                                                                 patience=cfg.optim.patience)
        else:
            assert cfg.optim.schedule == 'inverse_square_root_with_warmup', f'Unimplemented lr scheduler {cfg.optim.schedule}'
            bse_scheduler = InverseSquareRootWithWarmUpScheduler(lr=cfg.optim.lr_bse,
                                                                 warmup=cfg.optim.warmup,
                                                                 warmup_init_lr=cfg.optim.warmup_init_lr,
                                                                 min_lr=cfg.optim.min_lr)

    else:
        bse_scheduler = None

    return nmt_scheduler, bse_scheduler
