import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler

from pipnet.affine_ops import (
    affine,
    get_rotation_mtrx,
    get_zeros_mask,
    get_affine_match_mask,
    draw_angles,
)

from tqdm import tqdm
from typing import Dict, Tuple, Callable, Optional


def train_step_plain(
    network: torch.nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: Dict[str, float],
    pretrain: bool,
    finetune: bool,
    criterion: Callable,
    train_iter,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a standard PIP-Net train step.
    """
    # Perform a forward pass through the network;
    proto_features, pooled, logits = network(torch.cat([x1, x2]))

    # Calculate loss and metrics;
    norm_mul = network.module._classification.normalization_multiplier

    loss, acc = calculate_loss(
        mode="PLAIN",
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=None,
        loss_weights=loss_weights,
        net_normalization_multiplier=norm_mul,
        pretrain=pretrain,
        finetune=finetune,
        criterion=criterion,
        train_iter=train_iter,
        device=device,
        print=True,
        EPS=1e-7,
    )
    return loss, acc


def train_step_rot_inv(
    network: torch.nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: Dict[str, float],
    pretrain: bool,
    finetune: bool,
    criterion: Callable,
    train_iter,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs PIP-Net train step
    with forward and inverse rotation.
    """
    with torch.no_grad():
        # Randomly draw angles;
        batch_size = targets.shape[0]

        if pretrain:
            angles = draw_angles(batch_size, min_angle=-30, max_angle=30, step=5)
        else:
            angles = draw_angles(batch_size, min_angle=-10, max_angle=10, step=5)

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
    norm_mul = network.module._classification.normalization_multiplier

    loss, acc = calculate_loss(
        mode="INV",
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=loss_mask,
        loss_weights=loss_weights,
        net_normalization_multiplier=norm_mul,
        pretrain=pretrain,
        finetune=finetune,
        criterion=criterion,
        train_iter=train_iter,
        device=device,
        print=True,
        EPS=1e-7,
    )
    return loss, acc


def train_step_rot_match(
    network: torch.nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: Dict[str, float],
    pretrain: bool,
    finetune: bool,
    criterion: Callable,
    train_iter,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs PIP-Net train step
    with rotation and matching.
    """
    with torch.no_grad():
        # Randomly draw angles;
        batch_size = targets.shape[0]

        if pretrain:
            angles = draw_angles(batch_size, min_angle=-30, max_angle=30, step=5)
        else:
            angles = draw_angles(batch_size, min_angle=-10, max_angle=10, step=5)

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
    norm_mul = network.module._classification.normalization_multiplier

    loss, acc = calculate_loss(
        mode="MATCH",
        proto_features=proto_features,
        pooled=pooled,
        logits=logits,
        targets=targets,
        loss_mask=match_mtrx,
        loss_weights=loss_weights,
        net_normalization_multiplier=norm_mul,
        pretrain=pretrain,
        finetune=finetune,
        criterion=criterion,
        train_iter=train_iter,
        device=device,
        print=True,
        EPS=1e-7,
    )
    return loss, acc


def train_pipnet(
    net,
    train_loader,
    optimizer_net,
    optimizer_classifier,
    scheduler_net,
    scheduler_classifier,
    criterion,
    epoch,
    nr_epochs,
    device,
    pretrain=False,
    finetune=False,
    mode: str = "PLAIN",
    progress_prefix: str = 'Train Epoch',
):
    # Initialize gradient scaler for AMP
    scaler = GradScaler()

    # Make sure the model is in train mode
    net.train()

    print(f"Training mode: {mode}")
    print(f"Device: {device}")

    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + '%s' % epoch,
        mininterval=2.0,
        ncols=0,
    )

    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1
    print("Number of parameters that require gradient: ", count_param, flush=True)

    loss_weights = {}
    if pretrain:
        loss_weights["a_loss_w"] = epoch / nr_epochs
        loss_weights["t_loss_w"] = 5.0
        loss_weights["c_loss_w"] = 0.0
        loss_weights["u_loss_w"] = 0.5  # <- ignored
    else:
        loss_weights["a_loss_w"] = 5.0
        loss_weights["t_loss_w"] = 2.0
        loss_weights["c_loss_w"] = 2.0
        loss_weights["u_loss_w"] = 0.0  # <- ignored

    print(
        f"Align weight: {loss_weights['a_loss_w']}, ",
        f"Tanh weight:  {loss_weights['a_loss_w']}, ",
        f"Class weight: {loss_weights['c_loss_w']}",
        flush=True,
    )
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)

    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (x1, x2, y) in train_iter:

        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a train step
        with autocast():
            if mode == "MATCH":
                loss, acc = train_step_rot_match(
                    network=net,
                    x1=x1,
                    x2=x2,
                    targets=y,
                    loss_weights=loss_weights,
                    pretrain=pretrain,
                    finetune=finetune,
                    criterion=criterion,
                    train_iter=train_iter,
                    device=device,
                )
            elif mode == "INV":
                loss, acc = train_step_rot_inv(
                    network=net,
                    x1=x1,
                    x2=x2,
                    targets=y,
                    loss_weights=loss_weights,
                    pretrain=pretrain,
                    finetune=finetune,
                    criterion=criterion,
                    train_iter=train_iter,
                    device=device,
                )
            else:
                loss, acc = train_step_plain(
                    network=net,
                    x1=x1,
                    x2=x2,
                    targets=y,
                    loss_weights=loss_weights,
                    pretrain=pretrain,
                    finetune=finetune,
                    criterion=criterion,
                    train_iter=train_iter,
                    device=device,
                )

        # Compute the gradient
        scaler.scale(loss).backward()

        # Optimize
        if not pretrain:
            scaler.step(optimizer_classifier)
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            scaler.step(optimizer_net)
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)

        # Update gradient scaler
        scaler.update()

        with torch.no_grad():
            total_acc += acc.item()
            total_loss += loss.item()

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3,
                                                                    min=0.))  # set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(
                    torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0))
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))

    train_info['train_accuracy'] = total_acc / float(i + 1)
    train_info['loss'] = total_loss / float(i + 1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    return train_info


def calculate_loss(
    mode: str,
    proto_features: torch.Tensor,
    pooled: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_weights: Dict[str, float],
    net_normalization_multiplier: float,
    pretrain: bool,
    finetune: bool,
    criterion: Callable,
    train_iter,
    device: str,
    print: bool = True,
    EPS: float = 1e-7,
):
    a_loss_w = loss_weights["a_loss_w"]
    t_loss_w = loss_weights["t_loss_w"]
    c_loss_w = loss_weights["c_loss_w"]

    targets = torch.cat([targets, targets])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    # Calculate loss based on training mode;
    loss = torch.tensor(0.0).to(device)
    acc = torch.tensor(0.0).to(device)

    a1_loss = align_loss(pf1, pf2.detach(), loss_mask, mode)
    a2_loss = align_loss(pf2, pf1.detach(), loss_mask, mode)
    a_loss = (a1_loss + a2_loss) / 2.0
    t_loss = (tanh_loss(pooled1) + tanh_loss(pooled2)) / 2.0

    if not finetune:
        loss += a_loss_w * a_loss + t_loss_w * t_loss

    if not pretrain:
        c_loss = class_loss(
            logits=logits,
            targets=targets,
            norm_mul=net_normalization_multiplier,
            criterion=criterion,
        )
        loss += c_loss_w * c_loss
        acc = class_accuracy(logits, targets)

    if print:
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                    f'L: {loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{t_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}',
                    refresh=False,
                )
            elif finetune:
                train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{c_loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{t_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',
                    refresh=False,
                )
            else:
                train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{c_loss.item():.3f}, LA:{a_loss.item():.2f}, LT:{t_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',
                    refresh=False,
                )
    return loss, acc


def align_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    mask: Optional[torch.Tensor],
    mode: str = "PLAIN",
    EPS: float = 1e-7,
):
    assert z1.shape == z2.shape
    assert z2.requires_grad is False

    if mode == "MATCH":
        # Flatten H, W dims;
        N, D, _, _ = z1.shape
        z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
        z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)

        # Calculate inner product;
        x_inner = torch.bmm(z1, z2.transpose(1, 2))

        # Calculate masked loss;
        loss = -torch.log(x_inner + EPS)
        loss *= mask
        loss = loss.sum() / mask.sum()

    elif mode == "INV":
        # Flatten H, W dims;
        N, D, _, _ = z1.shape
        z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
        z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)
        mask = mask.permute(0, 2, 3, 1).reshape(N, -1)

        # Calculate inner product;
        x_inner = torch.sum(z1 * z2, dim=2)

        # Calculate masked loss;
        loss = -torch.log(x_inner + EPS)
        loss *= mask
        loss = loss.sum() / mask.sum()

    else:
        # from https://gitlab.com/mipl/carl/-/blob/main/losses.py
        z1 = z1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
        z2 = z2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
        loss = torch.einsum("nc,nc->n", [z1, z2])
        loss = -torch.log(loss + EPS).mean()

    return loss


def tanh_loss(inputs: torch.Tensor, EPS: float = 1e-7):
    loss = torch.tanh(torch.sum(inputs, dim=0))
    loss = -torch.log(loss + EPS).mean()
    return loss


def class_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    norm_mul: float,
    criterion: Callable,
) -> torch.Tensor:
    logits = torch.log1p(logits ** norm_mul)
    y_pred = F.log_softmax(logits, dim=1)
    return criterion(y_pred, targets)


def class_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(torch.eq(preds, targets))
    return correct / float(len(targets))


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want.
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss
