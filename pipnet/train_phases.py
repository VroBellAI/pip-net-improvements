import os
import torch
from torch import nn
import matplotlib.pyplot as plt

from pipnet.train import train_pipnet
from pipnet.test import eval_pipnet
from util.log import Log

from typing import List


def pretrain_self_supervised(
    num_epochs: int,
    network: nn.Module,
    backbone_optimizer: torch.optim.Optimizer,
    backbone_scheduler: torch.optim.lr_scheduler.LRScheduler,
    head_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader,
    log: Log,
    log_dir: str,
    train_mode: str,
    state_dict_dir_net: str,
    params_to_train: List[torch.Tensor],
    params_to_freeze: List[torch.Tensor],
    params_backbone: List[torch.Tensor],
    device: torch.device,
):
    lrs_pretrain_net = []
    for epoch in range(1, num_epochs + 1):
        for param in params_to_train:
            param.requires_grad = True
        for param in network.module._add_on.parameters():
            param.requires_grad = True
        for param in network.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            # Can be set to False when you want to freeze more layers
            param.requires_grad = True
        for param in params_backbone:
            # Can be set to True when you want to train whole backbone
            # (e.g. if dataset is very different from ImageNet)
            param.requires_grad = False

        print("\nPretrain Epoch", epoch, "with batch size", train_loader.batch_size, flush=True)

        # Pretrain prototypes
        train_info = train_pipnet(
            net=network,
            train_loader=train_loader,
            optimizer_net=backbone_optimizer,
            optimizer_classifier=head_optimizer,
            scheduler_net=backbone_scheduler,
            scheduler_classifier=None,
            criterion=criterion,
            epoch=epoch,
            nr_epochs=num_epochs,
            device=device,
            pretrain=True,
            finetune=False,
            mode=train_mode,
        )

        lrs_pretrain_net += train_info['lrs_net']

        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(log_dir, 'lr_pretrain_net.png'))

        log.log_values(
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

    if state_dict_dir_net == '':
        network.eval()
        torch.save(
            {
                'model_state_dict': network.state_dict(),
                'optimizer_net_state_dict': backbone_optimizer.state_dict()
            },
            os.path.join(os.path.join(log_dir, 'checkpoints'), 'net_pretrained')
        )
        network.train()


def train_supervised(
    num_epochs: int,
    pretrain_epochs: int,
    freeze_epochs: int,
    network: nn.Module,
    backbone_optimizer: torch.optim.Optimizer,
    backbone_scheduler: torch.optim.lr_scheduler.LRScheduler,
    head_optimizer: torch.optim.Optimizer,
    head_scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    train_loader,
    test_loader,
    bias: bool,
    log: Log,
    log_dir: str,
    train_mode: str,
    state_dict_dir_net: str,
    params_to_train: List[torch.Tensor],
    params_to_freeze: List[torch.Tensor],
    params_backbone: List[torch.Tensor],
    device: torch.device,
):
    for param in network.module.parameters():
        param.requires_grad = False

    for param in network.module._classification.parameters():
        param.requires_grad = True

    frozen = True
    lrs_net = []
    lrs_classifier = []

    # During finetuning, only train classification layer and freeze rest.
    # Usually done for a few epochs (at least 1, more depends on size of dataset).
    epochs_to_finetune = 3

    for epoch in range(1, num_epochs + 1):

        if epoch <= epochs_to_finetune and (pretrain_epochs > 0 or state_dict_dir_net != ''):
            for param in network.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False

            finetune = True

        else:
            finetune = False
            if frozen:
                # unfreeze backbone
                if epoch > (freeze_epochs):
                    for param in network.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True  # Can be set to False if you want to train fewer layers of backbone
                    for param in network.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False

        print("\n Epoch", epoch, "frozen:", frozen, flush=True)
        if (epoch == num_epochs or epoch % 30 == 0) and num_epochs > 1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                network.module._classification.weight.copy_(
                    torch.clamp(network.module._classification.weight.data - 0.001, min=0.))
                print("Classifier weights: ",
                      network.module._classification.weight[network.module._classification.weight.nonzero(as_tuple=True)], (
                      network.module._classification.weight[
                          network.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if bias:
                    print("Classifier bias: ", network.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(
            net=network,
            train_loader=train_loader,
            optimizer_net=backbone_optimizer,
            optimizer_classifier=head_optimizer,
            scheduler_net=backbone_scheduler,
            scheduler_classifier=head_scheduler,
            criterion=criterion,
            epoch=epoch,
            nr_epochs=num_epochs,
            device=device,
            pretrain=False,
            finetune=finetune,
            mode=train_mode,
        )

        lrs_net += train_info['lrs_net']
        lrs_classifier += train_info['lrs_class']

        # Evaluate model
        eval_pipnet(
            net=network,
            test_loader=test_loader,
            epoch=epoch,
            device=device,
            log=log,
            train_info=train_info,
        )

        with torch.no_grad():
            network.eval()
            torch.save(
                {
                    'model_state_dict': network.state_dict(),
                    'optimizer_net_state_dict': backbone_optimizer.state_dict(),
                    'optimizer_classifier_state_dict': head_optimizer.state_dict(),
                },
                os.path.join(os.path.join(log_dir, 'checkpoints'), 'net_trained'),
            )

            if epoch % 30 == 0:
                network.eval()
                torch.save(
                    {
                        'model_state_dict': network.state_dict(),
                        'optimizer_net_state_dict': backbone_optimizer.state_dict(),
                        'optimizer_classifier_state_dict': head_optimizer.state_dict(),
                    },
                    os.path.join(os.path.join(log_dir, 'checkpoints'), 'net_trained_%s' % str(epoch)),
                )

            # Plot learning rates;
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(log_dir, 'lr_net.png'))

            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(log_dir, 'lr_class.png'))

    network.eval()
    torch.save(
        {
            'model_state_dict': network.state_dict(),
            'optimizer_net_state_dict': backbone_optimizer.state_dict(),
            'optimizer_classifier_state_dict': head_optimizer.state_dict(),
        },
        os.path.join(os.path.join(log_dir, 'checkpoints'), 'net_trained_last'),
    )

