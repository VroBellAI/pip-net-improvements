import torch
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

from pipnet.pipnet import PIPNet
from pipnet.loss import WeightedSumLoss, LOSS_DATA
from pipnet.metrics import Metric
from pipnet.affine_ops import (
    affine,
    get_rotation_mtrx,
    get_zeros_mask,
    get_affine_match_mask,
    draw_angles,
)

from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Optional


def forward_train_plain(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs a standard PIP-Net train step.
    """
    # Perform a forward pass through the network;
    inputs = torch.cat([x1, x2])
    targets = torch.cat([targets, targets])
    model_output = network(inputs)

    # Calculate loss and metrics;
    loss_data = loss_fn(
        model_output=model_output,
        targets=targets,
        loss_mask=None,
    )
    metrics_data = {
        metric.name: metric(
            model_output=model_output,
            targets=targets,
        )
        for metric in metrics
    }

    return loss_data


def forward_train_rot_inv(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
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
    model_output = network(torch.cat([x_i, x_t]))

    # Rotate back the transformed feature mask;
    z_i, z_t = model_output.proto_feature_map.chunk(2)
    z_t = affine(z_t, t_inv_mtrx, padding_mode="zeros", device=device)

    with torch.no_grad():
        # Get zeros mask (to mask loss);
        # Each channel should be the same -> choose one;
        loss_mask = torch.ones_like(z_t).to(device)
        loss_mask = affine(loss_mask, t_inv_mtrx, padding_mode="zeros", device=device)
        loss_mask = get_zeros_mask(loss_mask)[:, 0:1, ...]
        loss_mask = loss_mask

    # Calculate loss and metrics;
    proto_feature_map = torch.cat([z_i, z_t])
    model_output.proto_feature_map = proto_feature_map

    # Calculate loss and metrics;
    loss_data = loss_fn(
        model_output=model_output,
        targets=targets,
        loss_mask=loss_mask,
    )
    return loss_data


def forward_train_rot_match(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
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
    model_output = network(torch.cat([x_i, x_t]))

    with torch.no_grad():
        # Generate coordinates match matrix;
        z_i, _ = model_output.proto_feature_map.chunk(2)
        match_mtrx = get_affine_match_mask(t_mtrx, z_i.shape, device)

    # Calculate loss and metrics;
    loss_data = loss_fn(
        model_output=model_output,
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
    optimizers: Dict[str, Optimizer],
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs train step with Full Precision.
    """
    # Reset the gradients;
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # Perform a train step;
    step_data = forward_pass_func(
        network=network,
        x1=x1,
        x2=x2,
        targets=targets,
        loss_func=loss_func,
        device=device,
    )

    # Compute the gradient;
    total_loss = step_data["total_loss"]
    total_loss.backward()

    # Optimize;
    for opt in optimizers.values():
        opt.step()

    return step_data


def train_step_amp(
    forward_pass_func: Callable,
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_func: PIPNetLoss,
    scaler: GradScaler,
    optimizers: Dict[str, Optimizer],
    device: torch.device,
) -> LOSS_DATA:
    """
    Performs train step with Automatic Mixed Precision.
    """
    # Reset the gradients;
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # Perform a train step;
    with autocast():
        step_data = forward_pass_func(
            network=network,
            x1=x1,
            x2=x2,
            targets=targets,
            loss_func=loss_func,
            device=device,
        )

    # Compute the gradient;
    total_loss = step_data["total_loss"]
    scaler.scale(total_loss).backward()

    # Optimize;
    for opt in optimizers.values():
        scaler.step(opt)

    # Update gradient scaler;
    scaler.update()
    return step_data


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


def step_data_to_str(loss_data: LOSS_DATA) -> str:
    """
    Converts loss data to string info.
    """
    loss_str = ""

    for loss in loss_data:
        loss_str += f"{loss}: {loss_data['loss'].item():.3f}"

    return loss_str

