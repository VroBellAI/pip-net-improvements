import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
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


class PIPNetOutput:
    def __init__(
        self,
        proto_feature_map: torch.Tensor,
        proto_feature_vec: torch.Tensor,
        logits: torch.Tensor,
        pre_softmax: torch.Tensor,
        softmax_out: torch.Tensor,
    ):
        self.proto_feature_map = proto_feature_map
        self.proto_feature_vec = proto_feature_vec
        self.logits = logits
        self.pre_softmax = pre_softmax
        self.softmax_out = softmax_out


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

    @property
    def params_classifier(self):
        return self._classification.parameters()

    @property
    def params_addon(self):
        return self._add_on.parameters()

    @property
    def params_to_train(self):
        return self._group_parameters('to_train')

    @property
    def params_to_freeze(self):
        return self._group_parameters('to_freeze')

    @property
    def params_backbone(self):
        return self._group_parameters('backbone')

    def forward(self, x: torch.Tensor, inference: bool = False) -> PIPNetOutput:
        features = self._net(x)
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)

        if inference:
            # During inference ignore prototypes
            # that have 0.1 similarity or lower;
            pooled = torch.where(pooled < 0.1, 0.0, pooled)

        logits = self._classification(pooled)
        pre_softmax = torch.log1p(logits ** self._multiplier)
        softmax_out = F.softmax(pre_softmax, dim=1)

        result = PIPNetOutput(
            proto_feature_map=proto_features,
            proto_feature_vec=pooled,
            logits=logits,
            pre_softmax=pre_softmax,
            softmax_out=softmax_out,
        )
        return result

    def get_class_weight(self) -> torch.Tensor:
        return self._classification.weight

    @torch.no_grad()
    def set_class_weight(self, w: torch.Tensor):
        self._classification.weight.copy_(w)

    def get_class_bias(self) -> torch.Tensor:
        return self._classification.bias

    @torch.no_grad()
    def set_class_bias(self, b: torch.Tensor):
        self._classification.bias.copy_(b)

    def get_norm_mul(self) -> torch.Tensor:
        return self._classification.normalization_multiplier

    @torch.no_grad()
    def set_norm_mul(self, norm_mul: torch.Tensor):
        self._classification.normalization_multiplier.copy_(norm_mul)

    @torch.no_grad()
    def get_nonzero_class_weight(self) -> torch.Tensor:
        class_w = self._classification.weight
        return class_w[class_w.nonzero(as_tuple=True)]

    def get_num_prototypes(self) -> int:
        return self._num_prototypes

    def get_num_classes(self) -> int:
        return self._num_classes

    def _group_parameters(self, group_name: str):
        # Define grouping conditions;
        convnext_conditions = {
            "to_train": [
                lambda name: "features.7.2" in name,
            ],
            "to_freeze": [
                lambda name: "features.7.2" not in name,
                lambda name: ("features.7" in name or "features.6" in name),
            ],
            "backbone": [
                lambda name: "features.7.2" not in name,
                lambda name: "features.7" not in name,
                lambda name: "features.6" not in name,
            ],
        }

        resnet_conditions = {
            "to_train": [
                lambda name: "layer4.2" in name,
            ],
            "to_freeze": [
                lambda name: "layer4.2" not in name,
                lambda name: ("layer4" in name or "layer3" in name),
            ],
            "backbone": [
                lambda name: "layer2" in name,
            ],
        }

        # Select conditions based on the chosen architecture;
        if "resnet50" in self._arch_name:
            print("chosen network is resnet50", flush=True)
            conditions = resnet_conditions

        elif "convnext" in self._arch_name:
            print("chosen network is convnext", flush=True)
            conditions = convnext_conditions
        else:
            raise Exception("Network is not ResNet50 or ConvNext!")

        # Params grouping generator;
        def params_generator():
            group_conditions = conditions[group_name]

            for name, param in self.named_parameters():
                if all([cond(name) for cond in group_conditions]):
                    yield param

        return params_generator()

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

    def count_gradient_params(self) -> int:
        grad_param_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad_param_count += 1
        return grad_param_count


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
    optimizers: Dict[str, Optimizer],
):
    checkpoints_dir = os.path.join(log_dir, 'checkpoints')
    save_dir = os.path.join(checkpoints_dir, checkpoint_name)

    network.eval()
    state_dict = {'model_state_dict': network.state_dict()}

    for opt_name, opt in optimizers.items():
        state_dict[opt_name] = opt.state_dict()

    torch.save(state_dict, save_dir)
    network.train()
