import torch
from torch import nn
from util.func import init_weights_xavier

from typing import Tuple, Dict


def initialize_head(network: nn.Module, bias: bool):
    torch.nn.init.normal_(network.module._classification.weight, mean=1.0, std=0.1)
    torch.nn.init.constant_(network.module._multiplier, val=2.0)

    if bias:
        torch.nn.init.constant_(network.module._classification.bias, val=0.0)

    print(
        "Classification layer initialized with mean",
        torch.mean(network.module._classification.weight).item(),
        flush=True,
    )


def initialize_model(network: nn.Module, bias: bool):
    network.module._add_on.apply(init_weights_xavier)
    initialize_head(network, bias)
    network.module._multiplier.requires_grad = False


def load_model(
    network: nn.Module,
    backbone_optimizer: nn.Module,
    head_optimizer: nn.Module,
    device: torch.device,
    num_prototypes: int,
    num_classes: int,
    state_dict_dir_net: str,
    bias: bool,
):
    # Load backbone;
    checkpoint = torch.load(state_dict_dir_net, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'], strict=True)
    network.module._multiplier.requires_grad = False
    print("Pretrained network loaded", flush=True)

    # Load backbone optimizer;
    try:
        backbone_optimizer.load_state_dict(checkpoint['optimizer_net_state_dict'])
        print("Pretrained backbone optimizer loaded", flush=True)
    except:
        pass

    # Optionally: re-initialize head;
    max_num_valid_class_w = 0.8 * (num_prototypes * num_classes)
    class_w_mean = torch.mean(network.module._classification.weight).item()
    num_valid_class_w = torch.count_nonzero(torch.relu(network.module._classification.weight - 1e-5)).float().item()

    if 1.0 < class_w_mean < 3.0 and num_valid_class_w > max_num_valid_class_w:
        print(
            "We assume that the classification layer is not yet trained. "
            "We re-initialize it...",
            flush=True,
        )
        initialize_head(network, bias)

    # else: #uncomment these lines if you want to load the optimizer too
    #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
    #         head_optimizer.load_state_dict(checkpoint['optimizer_classifier_state_dict'])


def initialize_or_load_model(
    network: nn.Module,
    backbone_optimizer: nn.Module,
    head_optimizer: nn.Module,
    num_classes: int,
    num_prototypes: int,
    state_dict_dir_net: str,
    bias: bool,
    device: torch.device,
):
    with torch.no_grad():
        if state_dict_dir_net == '':
            initialize_model(network, bias)
        else:
            load_model(
                network=network,
                backbone_optimizer=backbone_optimizer,
                head_optimizer=head_optimizer,
                state_dict_dir_net=state_dict_dir_net,
                bias=bias,
                device=device,
                num_prototypes=num_prototypes,
                num_classes=num_classes,
            )


@torch.no_grad()
def zero_out_irrelevant_protos(
    top_ks: Dict,
    network: nn.Module,
    min_score: float = 0.1,
):
    # Set weights of prototypes, that are never really found in projection, to 0;
    zeroed_proto_idxs = []
    head_mtrx = network.module._classification.weight
    if top_ks:
        for proto_idx in top_ks.keys():
            found = False

            for (_, score) in top_ks[proto_idx]:
                if score <= min_score:
                    continue

                found = True
                break

            if not found:
                torch.nn.init.zeros_(head_mtrx[:, proto_idx])
                zeroed_proto_idxs.append(proto_idx)

    print(
        f"Weights of prototypes {zeroed_proto_idxs}"
        f"are set to zero because it is never detected with similarity>0.1 in the training set",
        flush=True,
    )


def get_feature_vec_dim(network: nn.Module, train_loader, device: torch.device) -> Tuple[int, ...]:
    with torch.no_grad():
        xs1, _, _ = next(iter(train_loader))
        xs1 = xs1.to(device)
        proto_features, _, _ = network(xs1)
        return proto_features.shape
