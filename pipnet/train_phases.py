import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple

from pipnet import save_pipnet, PIPNet
from pipnet.loss import PIPNetLoss
from pipnet.train_steps import train_epoch
from pipnet.test import eval_pipnet
from util.logger import Logger, plot_learning_rate_curve
from util.func import connect_gradients, disconnect_gradients


def manage_pretrain_gradients(network: PIPNet):
    """
    Connects/disconnects params gradients
    for self-supervised pre-training phase.
    """
    connect_gradients(network.params_to_train)
    connect_gradients(network.params_add_on)

    # Can be set to disconnect when you want to freeze more layers;
    connect_gradients(network.params_to_freeze)

    # Can be set to connect when you want to train whole backbone
    # (e.g. if dataset is very different from ImageNet);
    disconnect_gradients(network.params_backbone)

    # Disable training of classification layer;
    disconnect_gradients(network.params_classifier)

    # TODO: redundant?
    network.module._classification.requires_grad = False


def pretrain_self_supervised(
    num_epochs: int,
    network: PIPNet,
    backbone_optimizer: Optimizer,
    backbone_scheduler: LRScheduler,
    loss_func: PIPNetLoss,
    train_loader,
    logger: Logger,
    state_dict_dir_net: str,
    device: torch.device,
    use_mixed_precision: bool,
):
    print(f"AMP: {use_mixed_precision}")

    # Init learning rate storage;
    lr_hist = {"backbone": []}

    # Set loss to pretrain mode;
    loss_func.set_pretrain_mode()
    loss_func.set_num_epochs(num_epochs)

    # Connect/disconnect gradients to specified params;
    manage_pretrain_gradients(network)

    for epoch in range(1, num_epochs + 1):
        print(
            f"\nPretrain Epoch {epoch} "
            f"with batch size {train_loader.batch_size}",
            flush=True,
        )

        # Track epoch index for loss weights calculation;
        loss_func.set_curr_epoch(epoch)

        # Pretrain prototypes;
        train_info = train_epoch(
            epoch_idx=epoch,
            net=network,
            train_loader=train_loader,
            backbone_optimizer=backbone_optimizer,
            head_optimizer=None,
            backbone_scheduler=backbone_scheduler,
            head_scheduler=None,
            loss_func=loss_func,
            device=device,
            progress_prefix='Pretrain Epoch',
            use_mixed_precision=use_mixed_precision,
        )

        # Save backbone optimizer learning rate;
        lr_hist["backbone"] += train_info['lrs_net']

        # Plot learning rates;
        plot_learning_rate_curve(
            log_dir=logger.log_dir,
            name='lr_pretrain_net',
            lr_vec=lr_hist["backbone"],
        )

        # Log loss value;
        logger.log_values(
            'log_epoch_overview',
            epoch,
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            train_info['loss'],
        )

    # Save trained params;
    if state_dict_dir_net == '':
        save_pipnet(
            log_dir=logger.log_dir,
            checkpoint_name='net_pretrained',
            network=network,
            backbone_optimizer=backbone_optimizer,
        )


def manage_train_gradients(
    network: PIPNet,
    epoch: int,
    epochs_to_finetune: int,
    frozen: bool,
    freeze_epochs: int,
    pretrained: bool,
    state_dict_dir_net: str,
) -> Tuple[bool, bool]:
    """
    Connects/disconnects params gradients
    based on frozen & finetune conditions.
    Returns new frozen and new finetune states.
    """
    finetune_conditions = [
        epoch <= epochs_to_finetune,
        (pretrained or state_dict_dir_net != ''),
    ]

    # Finetune config;
    if all(finetune_conditions):
        disconnect_gradients(network.params_add_on)
        disconnect_gradients(network.params_to_train)
        disconnect_gradients(network.params_to_freeze)
        disconnect_gradients(network.params_backbone)
        # TODO: is it redundant?
        network.module._classification.requires_grad = True
        return frozen, True

    # Not finetune configs;
    # Not finetune, not frozen;
    if not frozen:
        return False, False

    # Unfreeze backbone;
    if epoch > freeze_epochs:
        connect_gradients(network.params_add_on)
        connect_gradients(network.params_to_freeze)
        connect_gradients(network.params_to_train)
        connect_gradients(network.params_backbone)
        # TODO: is it redundant?
        network.module._classification.requires_grad = True
        return False, False

    # Freeze first layers of backbone, train rest;
    # Can be disconnected if you want to train fewer layers of backbone;
    connect_gradients(network.params_to_freeze)
    connect_gradients(network.params_add_on)
    connect_gradients(network.params_to_train)
    disconnect_gradients(network.params_backbone)
    # TODO: is it redundant?
    network.module._classification.requires_grad = True
    return True, False


def train_supervised(
    num_epochs: int,
    freeze_epochs: int,
    network: PIPNet,
    backbone_optimizer: Optimizer,
    backbone_scheduler: LRScheduler,
    head_optimizer: Optimizer,
    head_scheduler: LRScheduler,
    loss_func: PIPNetLoss,
    train_loader,
    test_loader,
    logger: Logger,
    state_dict_dir_net: str,
    device: torch.device,
    pretrained: bool,
    use_mixed_precision: bool,
    epochs_to_finetune: int = 3,
):
    print(f"AMP: {use_mixed_precision}")

    # Init learning rate storage;
    lr_hist = {"backbone": [], "head": []}

    # Set loss to train mode;
    loss_func.set_train_mode()
    loss_func.set_num_epochs(num_epochs)

    # Initially freeze backbone params;
    frozen = True

    for epoch in range(1, num_epochs + 1):
        # Track epoch index for loss weights calculation;
        loss_func.set_curr_epoch(epoch)

        # Dynamically connect/disconnect gradients;
        frozen, finetune = manage_train_gradients(
            network=network,
            epoch=epoch,
            epochs_to_finetune=epochs_to_finetune,
            frozen=frozen,
            freeze_epochs=freeze_epochs,
            pretrained=pretrained,
            state_dict_dir_net=state_dict_dir_net,
        )
        print(f"\n Epoch {epoch} frozen: {frozen}", flush=True)

        # Set loss to finetune mode if needed;
        loss_func.finetune = finetune

        # Set small weights to zero;
        if (epoch == num_epochs or epoch % 30 == 0) and num_epochs > 1:
            network.clip_class_params(
                zero_small_weights=True,
                clip_bias=False,
                clip_norm_mul=False,
                print_result=True,
            )

        # Train classifier;
        train_info = train_epoch(
            epoch_idx=epoch,
            net=network,
            train_loader=train_loader,
            backbone_optimizer=backbone_optimizer,
            head_optimizer=head_optimizer,
            backbone_scheduler=backbone_scheduler,
            head_scheduler=head_scheduler,
            loss_func=loss_func,
            device=device,
            progress_prefix='Train Epoch',
            use_mixed_precision=use_mixed_precision,
        )

        # Save optimizers learning rates;
        lr_hist["backbone"] += train_info['lrs_net']
        lr_hist["head"] += train_info['lrs_class']

        # Evaluate model
        eval_pipnet(
            net=network,
            test_loader=test_loader,
            epoch=epoch,
            device=device,
            log=logger,
            train_info=train_info,
        )

        # Save checkpoint;
        save_pipnet(
            log_dir=logger.log_dir,
            checkpoint_name='net_trained',
            network=network,
            backbone_optimizer=backbone_optimizer,
            head_optimizer=head_optimizer,
        )

        if epoch % 30 == 0:
            save_pipnet(
                log_dir=logger.log_dir,
                checkpoint_name='net_trained_%s' % str(epoch),
                network=network,
                backbone_optimizer=backbone_optimizer,
                head_optimizer=head_optimizer,
            )

        # Plot learning rates;
        plot_learning_rate_curve(
            log_dir=logger.log_dir,
            name='lr_net',
            lr_vec=lr_hist["backbone"],
        )
        plot_learning_rate_curve(
            log_dir=logger.log_dir,
            name='lr_class',
            lr_vec=lr_hist["head"],
        )

    # Save trained params;
    save_pipnet(
        log_dir=logger.log_dir,
        checkpoint_name='net_trained_last',
        network=network,
        backbone_optimizer=backbone_optimizer,
        head_optimizer=head_optimizer,
    )
