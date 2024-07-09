import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from pipnet import PIPNet
from pipnet.loss import PIPNetLoss, LOSS_DATA
from pipnet.affine_ops import (
    affine,
    get_rotation_mtrx,
    get_zeros_mask,
    get_affine_match_mask,
    draw_angles,
)

from tqdm import tqdm
from typing import Dict, Tuple, Callable, Optional


def forward_train_plain(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs a standard PIP-Net train step.
    """
    # Perform a forward pass through the network;
    proto_features, pooled, logits = network(torch.cat([x1, x2]))

    # Calculate loss and metrics;
    loss_data = loss_func(
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=None,
    )
    return loss_data


def forward_train_rot_inv(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs PIP-Net train step
    with forward and inverse rotation.
    """
    with torch.no_grad():
        # Randomly draw angles;
        batch_size = targets.shape[0]
        angles = draw_angles(batch_size, min_angle=-30, max_angle=30, step=2.5)

        # Get rotation matrix and inverse rotation matrix;
        t_mtrx = get_rotation_mtrx(angles).to(device)
        t_inv_mtrx = get_rotation_mtrx(-angles).to(device)

        # Transform;
        x_i = x1
        x_t = affine(x2, t_mtrx, padding_mode="reflection", device=device)

    # Forward pass;
    proto_features, pooled, logits = network(torch.cat([x_i, x_t]))

    # Rotate back the transformed feature mask;
    z_i, z_t = proto_features.chunk(2)
    z_t = affine(z_t, t_inv_mtrx, padding_mode="zeros", device=device)

    with torch.no_grad():
        # Get zeros mask (to mask loss);
        # Each channel should be the same -> choose one;
        loss_mask = torch.ones_like(z_t).to(device)
        loss_mask = affine(loss_mask, t_inv_mtrx, padding_mode="zeros", device=device)
        loss_mask = get_zeros_mask(loss_mask)[:, 0:1, ...]
        loss_mask = loss_mask

    # Calculate loss and metrics;
    proto_features = torch.cat([z_i, z_t])

    # Calculate loss and metrics;
    loss_data = loss_func(
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=loss_mask,
    )
    return loss_data


def forward_train_rot_match(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs PIP-Net train step
    with rotation and matching.
    """
    with torch.no_grad():
        # Randomly draw angles;
        batch_size = targets.shape[0]
        angles = draw_angles(batch_size, min_angle=-30, max_angle=30, step=2.5)

        # Get rotation matrix;
        t_mtrx = get_rotation_mtrx(angles).to(device)

        # Get identity and transformed tensor;
        x_i = x1
        x_t = affine(x2, t_mtrx, padding_mode="reflection", device=device)

    # Forward pass;
    proto_features, pooled, logits = network(torch.cat([x_i, x_t]))

    with torch.no_grad():
        # Generate coordinates match matrix;
        z_i, _ = proto_features.chunk(2)
        match_mtrx = get_affine_match_mask(t_mtrx, z_i.shape, device)

    # Calculate loss and metrics;
    loss_data = loss_func(
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=match_mtrx,
    )
    return loss_data


def train_step_fp(
    forward_pass_func: Callable,
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    scaler: Optional[GradScaler],
    head_optimizer: Optimizer,
    backbone_optimizer: Optimizer,
    device: torch.device,
    pretrain: bool,
    finetune: bool,
) -> LOSS_DATA:
    """
    Performs train step with Full Precision.
    """
    # Reset the gradients;
    head_optimizer.zero_grad(set_to_none=True)
    backbone_optimizer.zero_grad(set_to_none=True)

    # Perform a train step;
    loss_data = forward_pass_func(
        network=network,
        x1=x1,
        x2=x2,
        targets=targets,
        loss_func=loss_func,
        device=device,
    )

    # Compute the gradient;
    total_loss = loss_data["total_loss"]
    total_loss.backward()

    # Optimize;
    if not pretrain:
        head_optimizer.step()

    if not finetune:
        backbone_optimizer.step()

    return loss_data


def train_step_amp(
    forward_pass_func: Callable,
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    scaler: GradScaler,
    head_optimizer: Optimizer,
    backbone_optimizer: Optimizer,
    device: torch.device,
    pretrain: bool,
    finetune: bool,
) -> LOSS_DATA:
    """
    Performs train step with Automatic Mixed Precision.
    """
    # Reset the gradients;
    head_optimizer.zero_grad(set_to_none=True)
    backbone_optimizer.zero_grad(set_to_none=True)

    # Perform a train step;
    with autocast():
        loss_data = forward_pass_func(
            network=network,
            x1=x1,
            x2=x2,
            targets=targets,
            loss_func=loss_func,
            device=device,
        )

    # Compute the gradient;
    total_loss = loss_data["total_loss"]
    scaler.scale(total_loss).backward()

    # Optimize;
    if not pretrain:
        scaler.step(head_optimizer)

    if not finetune:
        scaler.step(backbone_optimizer)

    # Update gradient scaler;
    scaler.update()
    return loss_data


def select_train_step(
    use_mixed_precision: bool,
) -> Tuple[Callable, Optional[GradScaler]]:
    """
    Selects Full or Mixed precision train step function;
    """
    if use_mixed_precision:
        return train_step_amp, GradScaler()

    return train_step_fp, None


def select_forward_pass(mode: str) -> Callable:
    """
    Selects training forward pass function;
    """
    if mode == "MATCH":
        return forward_train_rot_match
    elif mode == "INV":
        return forward_train_rot_inv
    elif mode == "PLAIN":
        return forward_train_plain

    raise Exception(f"Training mode {mode} not implemented!")


def train_epoch(
    epoch_idx: int,
    net: PIPNet,
    train_loader: DataLoader,
    backbone_optimizer: Optimizer,
    head_optimizer: Optional[Optimizer],
    backbone_scheduler: LRScheduler,
    head_scheduler: Optional[LRScheduler],
    loss_func: PIPNetLoss,
    device: torch.device,
    use_mixed_precision: bool = True,
    progress_prefix: str = 'Train Epoch',
) -> Dict[str, float]:
    """
    Performs single train epoch.
    Returns train info:
    loss value, accuracy value, learning rates.
    """
    # Extract selected augmentation mode;
    aug_mode = loss_func.aug_mode
    print(f"Training aug mode: {aug_mode}")
    print(f"Device: {device}")

    # Select train step function (AMP or FP);
    train_step_func, scaler = select_train_step(use_mixed_precision)

    # Select forward pass function (Rotations aug or plain);
    forward_pass_func = select_forward_pass(aug_mode)

    # Make sure the model is in train mode;
    net.train()

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0
    num_steps = len(train_loader)

    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + '%s' % epoch_idx,
        mininterval=2.0,
        ncols=0,
    )

    # Count parameters that require gradient;
    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1

    print(
        f"Number of parameters that require gradient: {count_param}",
        flush=True,
    )

    print(
        f"Pretrain? {pretrain} Finetune? {finetune}",
        flush=True,
    )

    # Store learning rates values;
    lr_hist = {"backbone": [], "head": []}

    # Train epoch loop;
    for step_idx, (x1, x2, y) in train_iter:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        loss_data = train_step_func(
            forward_pass_func=forward_pass_func,
            network=net,
            x1=x1,
            x2=x2,
            targets=y,
            loss_func=loss_func,
            scaler=scaler,
            head_optimizer=head_optimizer,
            backbone_optimizer=backbone_optimizer,
            device=device,
            pretrain=pretrain,
            finetune=finetune,
        )

        # Print loss data;
        loss_info = loss_func.loss_data_to_str(loss_data)
        train_iter.set_postfix_str(
            s=loss_info,
            refresh=False,
        )

        # Set and save learning rates;
        if not pretrain:
            head_scheduler.step(epoch_idx - 1 + (step_idx / num_steps))
            lr_hist["head"].append(head_scheduler.get_last_lr()[0])

        if not finetune:
            backbone_scheduler.step()
            lr_hist["backbone"].append(backbone_scheduler.get_last_lr()[0])

        # Aggregate metrics;
        with torch.no_grad():
            total_acc += loss_data["acc"].item()
            total_loss += loss_data["total_loss"].item()

        # Clip classification parameters;
        if not pretrain:
            net.clip_class_params(
                zero_small_weights=True,
                clip_bias=True,
                clip_norm_mul=True,
                print_results=False,
            )

    train_info['train_accuracy'] = total_acc / float(step_idx + 1)
    train_info['loss'] = total_loss / float(step_idx + 1)
    train_info['lrs_net'] = lr_hist["backbone"]
    train_info['lrs_class'] = lr_hist["head"]
    return train_info

