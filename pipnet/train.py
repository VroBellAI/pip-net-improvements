import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from pipnet.pipnet import PIPNet, save_pipnet
from pipnet.test import evaluate_pipnet
from pipnet.loss import WeightedSumLoss
from pipnet.metrics import Metric, metric_data_to_str
from pipnet.affine_ops import (
    affine,
    get_rotation_mtrx,
    get_zeros_mask,
    get_affine_match_mask,
    draw_angles,
)
from util.logger import Logger
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
) -> Dict[str, torch.Tensor]:
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

    return loss_data | metrics_data


def forward_train_rot_inv(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
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
    inputs = torch.cat([x_i, x_t])
    targets = torch.cat([targets, targets])
    model_output = network(inputs)

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
    metrics_data = {
        metric.name: metric(
            model_output=model_output,
            targets=targets,
        )
        for metric in metrics
    }

    return loss_data | metrics_data


def forward_train_rot_match(
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
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
    inputs = torch.cat([x_i, x_t])
    targets = torch.cat([targets, targets])
    model_output = network(inputs)

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
    metrics_data = {
        metric.name: metric(
            model_output=model_output,
            targets=targets,
        )
        for metric in metrics
    }

    return loss_data | metrics_data


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

    raise Exception(
        f"Training augmentation mode {mode} not implemented!"
    )


def train_step_fp(
    forward_pass_func: Callable,
    network: PIPNet,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    scaler: Optional[GradScaler],
    optimizers: Dict[str, Optimizer],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Performs train step with Full Precision.
    """
    # Reset the gradients;
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    # Perform a train step;
    step_data = forward_pass_func(
        network=network,
        x1=x1,
        x2=x2,
        targets=targets,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
    )

    # Compute the gradients;
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
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    scaler: GradScaler,
    optimizers: Dict[str, Optimizer],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Performs train step with Automatic Mixed Precision.
    """
    # Reset the gradients;
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)

    # Perform a train step;
    with autocast():
        step_data = forward_pass_func(
            network=network,
            x1=x1,
            x2=x2,
            targets=targets,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device,
        )

    # Compute the gradients;
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


def print_epoch_info(
    epoch_idx: int,
    num_epochs: int,
    phase: str,
    batch_size: int,
    network: PIPNet,
    loss_fn: WeightedSumLoss,
    aug_mode: str,
    use_mixed_precision: bool,
    device: torch.device,
):
    print(f"\nEpoch: {epoch_idx}", flush=True)
    print(f"Phase: {phase}", flush=True)
    print(f"Num phase epochs: {num_epochs}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Aug: {aug_mode}", flush=True)
    print(f"AMP: {use_mixed_precision}", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        f"Number of parameters that require gradient: "
        f"{network.module.count_gradient_params()}",
        flush=True,
    )
    print("Partial Losses weights:", flush=True)
    print(
        ", ".join(
            f"{p_loss.name}: {p_loss.get_weight(epoch_idx=epoch_idx, num_epochs=num_epochs)}"
            for p_loss in loss_fn.partial_losses
        ),
        flush=True,
    )


def train_epoch(
    phase: str,
    epoch_idx: int,
    num_epochs: int,
    network: PIPNet,
    train_loader: DataLoader,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    loss_fn: WeightedSumLoss,
    metrics: List[Metric],
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
) -> Dict[str, float]:
    """
    Performs single train epoch.
    """
    # Print epoch info;
    print_epoch_info(
        epoch_idx=epoch_idx,
        num_epochs=num_epochs,
        phase=phase,
        batch_size=train_loader.batch_size,
        network=network,
        loss_fn=loss_fn,
        aug_mode=aug_mode,
        use_mixed_precision=use_mixed_precision,
        device=device,
    )

    # Select train step function (AMP or FP);
    train_step_func, scaler = select_train_step(use_mixed_precision)

    # Select forward pass function (Rotations aug or plain);
    forward_pass_func = select_forward_pass(aug_mode)

    # Make sure the model is in train mode;
    network.train()

    # Reset loss and metrics aggregation;
    loss_fn.reset()
    [metric.reset() for metric in metrics]

    # Save num steps;
    num_steps = len(train_loader)

    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"{phase} ep{epoch_idx}",
        mininterval=2.0,
        ncols=0,
    )

    # Store learning rates values;
    lr_hist = {lr_name: [] for lr_name in schedulers}

    # Train epoch loop;
    for step_idx, (x1, x2, y) in train_iter:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        step_data = train_step_func(
            forward_pass_func=forward_pass_func,
            network=network,
            x1=x1,
            x2=x2,
            targets=y,
            loss_fn=loss_fn,
            metrics=metrics,
            scaler=scaler,
            optimizers=optimizers,
            device=device,
        )

        # Set and save learning rates;
        for lr_sch_name, lr_sch in schedulers.items():

            if isinstance(lr_sch, CosineAnnealingWarmRestarts):
                step_val = epoch_idx - 1 + (step_idx / num_steps)
            else:
                step_val = None

            lr_sch.step(step_val)
            lr_hist[lr_sch_name].append(lr_sch.get_last_lr()[0])

        # Print loss data;
        train_iter.set_postfix_str(
            s=metric_data_to_str(step_data),
            refresh=False,
        )

        # Clip classification parameters;
        # TODO: clip by "requires grad..."
        if phase != "pretrain":
            network.module.clip_class_params(
                zero_small_weights=True,
                clip_bias=True,
                clip_norm_mul=True,
                print_results=False,
            )

    # Save the averaged epoch info;
    epoch_info = {
        "epoch_idx": epoch_idx,
        "phase": phase,
        "loss": loss_fn.get_aggregated_value(),
    }

    for metric in metrics:
        epoch_info[metric.name] = metric.get_aggregated_value()

    # TODO: bring back visualization;
    # for lr_name, lr_vec in lr_hist.items():
    #     epoch_info[f"LR_{lr_name}"] = lr_vec

    return epoch_info


def train_loop(
    num_epochs: int,
    init_epoch: int,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    network: PIPNet,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    loss_fn: WeightedSumLoss,
    train_metrics: List[Metric],
    test_metrics: List[Metric],
    logger: Logger,
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
    phase: str,
    save_period: int = 30,
):

    for epoch in range(init_epoch, num_epochs+init_epoch):
        # Track epochs with loss function;
        loss_fn.set_curr_epoch(epoch)

        # Track selected epochs;
        is_save_epoch = epoch % save_period == 0
        is_last_epoch = epoch == num_epochs

        # Set small class weights to zero;
        # TODO: clip by gradients req!
        if all([
            (is_last_epoch or is_save_epoch),
            num_epochs > 1,
            phase != "pretrain",
        ]):
            network.module.clip_class_params(
                zero_small_weights=True,
                clip_bias=False,
                clip_norm_mul=False,
                print_results=True,
            )

        # Train network;
        epoch_info = train_epoch(
            epoch_idx=epoch,
            num_epochs=num_epochs,
            network=network,
            train_loader=train_loader,
            optimizers=optimizers,
            schedulers=schedulers,
            loss_fn=loss_fn,
            metrics=train_metrics,
            device=device,
            aug_mode=aug_mode,
            use_mixed_precision=use_mixed_precision,
            phase=phase,
        )

        # Evaluate model;
        if test_loader:
            eval_info = evaluate_pipnet(
                network=network,
                test_loader=test_loader,
                metrics=test_metrics,
                epoch_idx=epoch,
                device=device,
            )
            epoch_info = epoch_info | eval_info

        # Log values;
        logger.log_epoch_info(epoch_info)

        # Save checkpoint;
        save_pipnet(
            log_dir=logger.log_dir,
            checkpoint_name=f'net_{phase}',
            network=network,
            optimizers=optimizers,
        )

        if is_save_epoch:
            save_pipnet(
                log_dir=logger.log_dir,
                checkpoint_name=f'net_{phase}_ep{epoch}',
                network=network,
                optimizers=optimizers,
            )
