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
from typing import Optional


class PIPNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        num_features: int,
        feature_net: nn.Module,
        add_on_layers: nn.Module,
        pool_layer: nn.Module,
        classification_layer: nn.Module
    ):
        super().__init__()
        self._num_features = num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

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
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        num_features=num_features,
        feature_net=feature_net,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    return pipnet
