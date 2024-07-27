import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from pipnet.pipnet import PIPNet
from pipnet.metrics import (
    ClassAccuracy,
    NumRelevantScores,
    NumAbstainedPredictions,
    ANZProto,
    ANZSimScores,
    LocalSize,
    NumNonZeroPrototypes,
    TopKClassAccuracy,
)
from pipnet.loss import (
    WeightedSumLoss,
    ClassLoss,
    TanhLoss,
    get_align_loss,
)
from pipnet.train import train_loop

from util.logger import Logger
from util.func import connect_gradients, disconnect_gradients

from typing import Dict


def pretrain(
    num_epochs: int,
    init_epoch: int,
    train_loader: DataLoader,
    network: PIPNet,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    logger: Logger,
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
    save_period: int,
):
    # Manage gradients;
    connect_gradients(network.module.params_to_train)
    connect_gradients(network.module.params_addon)

    # Can be set to disconnect when you want to freeze more layers;
    connect_gradients(network.module.params_to_freeze)

    # Can be set to connect when you want to train whole backbone
    # (e.g. if dataset is very different from ImageNet);
    disconnect_gradients(network.module.params_backbone)

    # Disable training of classification layer;
    disconnect_gradients(network.module.params_classifier)

    # Numerical stability constant;
    eps = 1e-7 if use_mixed_precision else 1e-10

    # Define losses;
    partial_losses = [
        TanhLoss(weight=5.0, device=device, eps=eps),
        # Set alignment weight to None
        # to dynamically modify it during training.
        get_align_loss(aug_mode)(weight=None, device=device, eps=eps),
    ]
    loss_fn = WeightedSumLoss(
        partial_losses=partial_losses,
        num_epochs=num_epochs,
        device=device,
    )

    # Define metrics;
    train_metrics = []
    test_metrics = []

    # Train network;
    train_loop(
        num_epochs=num_epochs,
        init_epoch=init_epoch,
        train_loader=train_loader,
        test_loader=None,
        network=network,
        optimizers=optimizers,
        schedulers=schedulers,
        loss_fn=loss_fn,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        logger=logger,
        device=device,
        aug_mode=aug_mode,
        use_mixed_precision=use_mixed_precision,
        phase="pretrain",
        save_period=save_period,
    )


def train_finetune(
    num_epochs: int,
    init_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    network: PIPNet,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    logger: Logger,
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
    save_period: int,
):
    # Manage gradients;
    disconnect_gradients(network.module.params_addon)
    disconnect_gradients(network.module.params_to_freeze)
    disconnect_gradients(network.module.params_to_train)
    disconnect_gradients(network.module.params_backbone)
    connect_gradients(network.module.params_classifier)

    # Define losses;
    partial_losses = [ClassLoss(weight=2.0, device=device)]
    loss_fn = WeightedSumLoss(
        partial_losses=partial_losses,
        num_epochs=num_epochs,
        device=device,
    )

    # Define metrics;
    train_metrics = [
        ClassAccuracy(),
        NumRelevantScores(thresh=0.1),
    ]
    test_metrics = [
        NumAbstainedPredictions(),
        ANZProto(),
        ANZSimScores(network=network),
        LocalSize(network=network),
        NumNonZeroPrototypes(network=network),
        TopKClassAccuracy(k=1),
        TopKClassAccuracy(k=5),
    ]

    # Train network;
    train_loop(
        num_epochs=num_epochs,
        init_epoch=init_epoch,
        train_loader=train_loader,
        test_loader=test_loader,
        network=network,
        optimizers=optimizers,
        schedulers=schedulers,
        loss_fn=loss_fn,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        logger=logger,
        device=device,
        aug_mode=aug_mode,
        use_mixed_precision=use_mixed_precision,
        phase="train_finetune",
        save_period=save_period,
    )


def train_frozen(
    num_epochs: int,
    init_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    network: PIPNet,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    logger: Logger,
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
    save_period: int,
):
    # Freeze first layers of backbone, train rest;
    # Can be disconnected if you want to train fewer layers of backbone;
    connect_gradients(network.module.params_to_freeze)
    connect_gradients(network.module.params_addon)
    connect_gradients(network.module.params_to_train)
    disconnect_gradients(network.module.params_backbone)
    connect_gradients(network.module.params_classifier)

    # Numerical stability constant;
    eps = 1e-7 if use_mixed_precision else 1e-10

    # Define losses;
    partial_losses = [
        ClassLoss(weight=2.0, device=device),
        TanhLoss(weight=2.0, device=device, eps=eps),
        get_align_loss(aug_mode)(weight=5.0, device=device, eps=eps),
    ]
    loss_fn = WeightedSumLoss(
        partial_losses=partial_losses,
        num_epochs=num_epochs,
        device=device,
    )

    # Define metrics;
    train_metrics = [
        ClassAccuracy(),
        NumRelevantScores(thresh=0.1),
    ]
    test_metrics = [
        NumAbstainedPredictions(),
        ANZProto(),
        ANZSimScores(network=network),
        LocalSize(network=network),
        NumNonZeroPrototypes(network=network),
        TopKClassAccuracy(k=1),
        TopKClassAccuracy(k=5),
    ]

    # Train network;
    train_loop(
        num_epochs=num_epochs,
        init_epoch=init_epoch,
        train_loader=train_loader,
        test_loader=test_loader,
        network=network,
        optimizers=optimizers,
        schedulers=schedulers,
        loss_fn=loss_fn,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        logger=logger,
        device=device,
        aug_mode=aug_mode,
        use_mixed_precision=use_mixed_precision,
        phase="train_frozen",
        save_period=save_period,
    )


def train_full(
    num_epochs: int,
    init_epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    network: PIPNet,
    optimizers: Dict[str, Optimizer],
    schedulers: Dict[str, LRScheduler],
    logger: Logger,
    device: torch.device,
    aug_mode: str,
    use_mixed_precision: bool,
    save_period: int,
):
    connect_gradients(network.module.params_addon)
    connect_gradients(network.module.params_to_freeze)
    connect_gradients(network.module.params_to_train)
    connect_gradients(network.module.params_backbone)
    connect_gradients(network.module.params_classifier)

    # Numerical stability constant;
    eps = 1e-7 if use_mixed_precision else 1e-10

    # Define losses;
    partial_losses = [
        ClassLoss(weight=2.0, device=device),
        TanhLoss(weight=2.0, device=device, eps=eps),
        get_align_loss(aug_mode)(weight=5.0, device=device, eps=eps),
    ]
    loss_fn = WeightedSumLoss(
        partial_losses=partial_losses,
        num_epochs=num_epochs,
        device=device,
    )

    # Define metrics;
    train_metrics = [
        ClassAccuracy(),
        NumRelevantScores(thresh=0.1),
    ]
    test_metrics = [
        NumAbstainedPredictions(),
        ANZProto(),
        ANZSimScores(network=network),
        LocalSize(network=network),
        NumNonZeroPrototypes(network=network),
        TopKClassAccuracy(k=1),
        TopKClassAccuracy(k=5),
    ]

    # Train network;
    train_loop(
        num_epochs=num_epochs,
        init_epoch=init_epoch,
        train_loader=train_loader,
        test_loader=test_loader,
        network=network,
        optimizers=optimizers,
        schedulers=schedulers,
        loss_fn=loss_fn,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        logger=logger,
        device=device,
        aug_mode=aug_mode,
        use_mixed_precision=use_mixed_precision,
        phase="train_full",
        save_period=save_period,
    )
