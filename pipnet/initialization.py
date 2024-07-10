import torch
from torch import nn
from torch.optim import Optimizer
from util.func import init_weights_xavier
from typing import Optional


@torch.no_grad()
def initialize_head(network: nn.Module):
    class_w = network.module.get_class_weight()
    class_b = network.module.get_class_bias()
    norm_mul = network.module.get_norm_mul()

    torch.nn.init.normal_(class_w, mean=1.0, std=0.1)
    torch.nn.init.constant_(norm_mul, val=2.0)

    if class_b is not None:
        torch.nn.init.constant_(class_b, val=0.0)

    print(
        "Classification layer initialized with mean",
        torch.mean(class_w).item(),
        flush=True,
    )


def initialize_model(network: nn.Module):
    network.module._add_on.apply(init_weights_xavier)
    initialize_head(network)
    norm_mul = network.module.get_norm_mul()
    norm_mul.requires_grad = False


@torch.no_grad()
def load_model(
    network: nn.Module,
    backbone_optimizer: Optimizer,
    head_optimizer: Optional[Optimizer],
    device: torch.device,
    state_dict_dir_net: str,
):
    # Load backbone;
    checkpoint = torch.load(state_dict_dir_net, map_location=device)
    network.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
    network.module.get_norm_mul().requires_grad = False
    print("Pretrained network loaded", flush=True)

    # Load backbone optimizer;
    try:
        backbone_optimizer.load_state_dict(checkpoint['optimizer_net_state_dict'])
        print("Pretrained backbone optimizer loaded", flush=True)
    except:
        pass

    # Optionally: re-initialize head;
    num_prototypes = network.module.get_num_prototypes()
    num_classes = network.module.get_num_classes()
    class_w = network.module.get_class_weight()

    max_num_valid_class_w = 0.8 * (num_prototypes * num_classes)
    class_w_mean = torch.mean(class_w).item()
    num_valid_class_w = torch.count_nonzero(torch.relu(class_w - 1e-5)).float().item()

    if 1.0 < class_w_mean < 3.0 and num_valid_class_w > max_num_valid_class_w:
        print(
            "We assume that the classification layer is not yet trained. "
            "We re-initialize it...",
            flush=True,
        )
        initialize_head(network)

    # Uncomment these lines if you want to load the head optimizer;
    # if 'optimizer_classifier_state_dict' in checkpoint.keys() and head_optimizer is not None:
    #     head_optimizer.load_state_dict(checkpoint['optimizer_classifier_state_dict'])


def initialize_or_load_model(
    network: nn.Module,
    backbone_optimizer: Optimizer,
    state_dict_dir_net: str,
    device: torch.device,
    head_optimizer: Optional[Optimizer] = None,
):
    with torch.no_grad():
        if state_dict_dir_net == '':
            initialize_model(network)
        else:
            load_model(
                network=network,
                backbone_optimizer=backbone_optimizer,
                head_optimizer=head_optimizer,
                state_dict_dir_net=state_dict_dir_net,
                device=device,
            )
