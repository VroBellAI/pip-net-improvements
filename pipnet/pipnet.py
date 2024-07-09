import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import (
    resnet18_features,
    resnet34_features,
    resnet50_features,
    resnet50_features_inat,
    resnet101_features,
    resnet152_features,
)
from features.convnext_features import (
    convnext_tiny_26_features,
    convnext_tiny_13_features,
)
from typing import List, Dict, Optional


class PIPNet(nn.Module):
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        num_prototypes: int,
        num_features: int,
        feature_net: nn.Module,
        add_on_layers: nn.Module,
        pool_layer: nn.Module,
        classification_layer: nn.Module
    ):
        super().__init__()
        self._arch_name = arch_name
        self._num_features = num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

        # Group network parameters;
        self.params_to_train = []
        self.params_to_freeze = []
        self.params_backbone = []
        self.params_add_on = self._add_on.parameters()
        self.params_classifier = self._classification.parameters()
        self.group_parameters()

    def forward(self, x: torch.Tensor, inference: bool = False):
        features = self._net(x)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)

        if inference:
            # During inference ignore prototypes
            # that have 0.1 similarity or lower;
            pooled = torch.where(pooled < 0.1, 0.0, pooled)

        # Out shape: (bs*2, num_classes)
        out = self._classification(pooled)
        return proto_features, pooled, out

    def get_class_weight(self) -> torch.Tensor:
        return self.module._classification.weight

    @torch.no_grad()
    def set_class_weight(self, w: torch.Tensor):
        self.module._classification.weight.copy_(w)

    def get_class_bias(self) -> torch.Tensor:
        return self.module._classification.bias

    @torch.no_grad()
    def set_class_bias(self, b: torch.Tensor):
        self.module._classification.bias.copy_(b)

    def get_norm_mul(self) -> torch.Tensor:
        return self.module._classification.normalization_multiplier

    @torch.no_grad()
    def set_norm_mul(self, norm_mul: torch.Tensor):
        self.module._classification.normalization_multiplier.copy_(norm_mul)

    @torch.no_grad()
    def get_nonzero_class_weight(self) -> torch.Tensor:
        class_w = self.self.module._classification.weight
        return class_w[class_w.nonzero(as_tuple=True)]

    def get_num_prototypes(self) -> int:
        return self.module._num_prototypes

    def get_num_classes(self) -> int:
        return self.module._num_classes

    def group_parameters(self):
        # set up optimizer
        if 'resnet50' in self._arch_name:
            # freeze resnet50 except last convolutional layer
            for name, param in self.module._net.named_parameters():
                if 'layer4.2' in name:
                    self.params_to_train.append(param)
                elif 'layer4' in name or 'layer3' in name:
                    self.params_to_freeze.append(param)
                elif 'layer2' in name:
                    self.params_backbone.append(param)
                else:
                    # Such that model training fits on one gpu.
                    param.requires_grad = False
                    # params_backbone.append(param)
            return

        if 'convnext' in self._arch_name:
            print("chosen network is convnext", flush=True)
            for name, param in self.module._net.named_parameters():
                if 'features.7.2' in name:
                    self.params_to_train.append(param)
                elif 'features.7' in name or 'features.6' in name:
                    self.params_to_freeze.append(param)
                # CUDA MEMORY ISSUES? COMMENT ABOVE LINES AND USE THE FOLLOWING INSTEAD
                # elif 'features.5' in name or 'features.4' in name:
                #     params_backbone.append(param)
                # else:
                #     param.requires_grad = False
                else:
                    self.params_backbone.append(param)
            return

        raise Exception("Network is not ResNet or ConvNext!")

    @torch.no_grad()
    def clip_class_params(
        self,
        zero_small_weights: bool = True,
        clip_bias: bool = False,
        clip_norm_mul: bool = False,
        print_results: bool = True,
    ):
        # Set small weights to zero;
        if zero_small_weights:
            class_w = self.get_class_weight()
            class_w_nonzero = torch.clamp(class_w.data - 1e-3, min=0.0)
            self.set_class_weight(class_w_nonzero)

        # Clip bias values to non-negative;
        class_b = self.get_class_bias()
        if clip_bias and class_b is not None:
            class_b_nonzero = torch.clamp(class_b.data, min=0.0)
            self.set_class_bias(class_b_nonzero)

        # Clip normalization multiplier;
        if clip_norm_mul:
            norm_mul = self.get_norm_mul()
            norm_mul_clipped = torch.clamp(norm_mul.data, min=0.0)
            self.set_norm_mul(norm_mul_clipped)

        if not print_results:
            return

        # Print results;
        torch.set_printoptions(profile="full")

        class_w_nonzero = self.get_nonzero_class_weight()
        print(
            f"Classifier weights: {class_w_nonzero} "
            f"{class_w_nonzero.shape}",
            flush=True,
        )

        class_b = self.get_class_bias()
        if class_b is not None:
            print(
                f"Classifier bias: {class_b}",
                flush=True,
            )

        norm_mul = self.get_norm_mul()
        print(
            f"Normalization multiplier: {norm_mul}",
            flush=True,
        )

        torch.set_printoptions(profile="default")

    @torch.no_grad()
    def zero_out_irrelevant_protos(
        self,
        top_ks: Dict,
        min_score: float = 0.1,
    ) -> List[int]:
        # Set weights of prototypes,
        # that are rarely found in projection, to 0;
        zeroed_proto_idxs = []
        head_mtrx = self.get_class_weight()

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

        return zeroed_proto_idxs


base_architecture_to_features = {
    'resnet18': resnet18_features,
    'resnet34': resnet34_features,
    'resnet50': resnet50_features,
    'resnet50_inat': resnet50_features_inat,
    'resnet101': resnet101_features,
    'resnet152': resnet152_features,
    'convnext_tiny_26': convnext_tiny_26_features,
    'convnext_tiny_13': convnext_tiny_13_features,
}


# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """
    Applies a linear transformation to the incoming data with non-negative weights.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
        )
        self.normalization_multiplier = nn.Parameter(
            torch.ones((1,), requires_grad=True),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, torch.relu(self.weight), self.bias)


def get_network(
    num_classes: int,
    num_features: int,
    backbone_name: str,
    pretrained: bool,
    bias: bool,
):
    backbone = base_architecture_to_features[backbone_name](pretrained=pretrained)
    features_name = str(backbone).upper()

    if 'next' in backbone_name:
        features_name = str(backbone_name).upper()

    if not features_name.startswith('RES') and not features_name.startswith('CONVNEXT'):
        raise Exception('other base architecture NOT implemented')

    conv_layers = [
        module for module in backbone.modules()
        if isinstance(module, nn.Conv2d)
    ]
    add_on_in_channels = conv_layers[-1].out_channels

    if num_features == 0:
        num_prototypes = add_on_in_channels
        print(f"Number of prototypes: {num_prototypes}", flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1),
        )
    else:
        num_prototypes = num_features
        print(
            f"Number of prototypes set "
            f"from {add_on_in_channels} to {num_prototypes}. "
            f"Extra 1x1 conv layer added. Not recommended.",
            flush=True,
        )
        add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=add_on_in_channels,
                out_channels=num_prototypes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Softmax(dim=1),
        )

    pool_layer = nn.Sequential(
        nn.AdaptiveMaxPool2d(output_size=(1, 1)),  # outputs (bs, ps, 1, 1);
        nn.Flatten()  # outputs (bs, ps);
    )
    
    class_layer = NonNegLinear(num_prototypes, num_classes, bias=bias)
    return backbone, add_on_layers, pool_layer, class_layer, num_prototypes


def get_pipnet(
    num_classes: int,
    num_features: int,
    backbone_name: str,
    pretrained: bool,
    bias: bool,
) -> PIPNet:
    (
        feature_net,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes
    ) = get_network(
        num_classes=num_classes,
        num_features=num_features,
        backbone_name=backbone_name,
        pretrained=pretrained,
        bias=bias,
    )

    pipnet = PIPNet(
        arch_name=backbone_name,
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        num_features=num_features,
        feature_net=feature_net,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    return pipnet


@torch.no_grad()
def save_pipnet(
    log_dir: str,
    checkpoint_name: str,
    network: torch.nn.Module,
    backbone_optimizer: Optional[torch.optim.Optimizer] = None,
    head_optimizer: Optional[torch.optim.Optimizer] = None,
):
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    save_dir = os.path.join(checkpoints_dir, checkpoint_name)

    network.eval()
    state_dict = {'model_state_dict': network.state_dict()}

    if backbone_optimizer is not None:
        state_dict['optimizer_net_state_dict'] = backbone_optimizer.state_dict()

    if head_optimizer is not None:
        state_dict['optimizer_classifier_state_dict'] = head_optimizer.state_dict()

    torch.save(state_dict, save_dir)
    network.train()
