import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_backbone_optimizer(
    optimizer_name: str,
    network: torch.nn.Module,
    lr_network: float,
    lr_backbone: float,
    lr_block: float,
    weight_decay: float,
) -> Optimizer:
    """
    Returns network backbone optimizer for pre-training and training phase.
    """
    backbone_params = [
        {"params": network.module.get_params_backbone(), "lr": lr_backbone, "weight_decay_rate": weight_decay},
        {"params": network.module.get_params_to_freeze(), "lr": lr_block, "weight_decay_rate": weight_decay},
        {"params": network.module.get_params_to_train(), "lr": lr_block, "weight_decay_rate": weight_decay},
        {"params": network.module.get_params_addon(), "lr": lr_block * 10.0, "weight_decay_rate": weight_decay},
    ]

    if optimizer_name != 'Adam':
        raise ValueError("this optimizer type is not implemented")

    optimizer = torch.optim.AdamW(
        backbone_params,
        lr=lr_network,
        weight_decay=weight_decay,
    )
    return optimizer


def get_head_optimizer(
    optimizer_name: str,
    network: torch.nn.Module,
    lr_network: float,
    weight_decay: float,
) -> Optimizer:
    """
    Returns network head optimizer for training phase.
    """
    network.get_norm_mul().requires_grad = False
    class_w = network.module.get_class_weight()
    class_b = network.module.get_class_bias()

    class_w_params = [class_w]
    class_b_params = [class_b] if class_b is not None else []

    head_params = [
        {"params": class_w_params, "lr": lr_network, "weight_decay_rate": weight_decay},
        {"params": class_b_params, "lr": lr_network, "weight_decay_rate": 0},
    ]

    if optimizer_name != 'Adam':
        raise ValueError("this optimizer type is not implemented")

    optimizer = torch.optim.AdamW(
        head_params,
        lr=lr_network,
        weight_decay=weight_decay,
    )
    return optimizer


def get_backbone_scheduler(
    backbone_optimizer: Optimizer,
    num_epochs: int,
    num_steps_per_epoch: int,
    eta_min: float,
) -> LRScheduler:
    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        backbone_optimizer,
        T_max=num_steps_per_epoch * num_epochs,
        eta_min=eta_min,
    )
    return backbone_scheduler


def get_head_scheduler(
    head_optimizer: Optimizer,
    num_epochs: int,
    eta_min: float,
) -> LRScheduler:
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        head_optimizer,
        T_0=5 if num_epochs <= 30 else 10,
        eta_min=eta_min,
        T_mult=1,
        verbose=False,
    )
    return head_scheduler
