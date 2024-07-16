import torch
import torch.nn.functional as F
from pipnet.pipnet import PIPNetOutput
from typing import List, Dict, Optional, Callable


class PartialLoss(torch.nn.Module):
    def __init__(
        self,
        name: str,
        weight: Optional[float],
        device: torch.device,
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.device = device

    def get_weight(self, **kwargs) -> float:
        return self.weight

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass


class ClassLoss(PartialLoss):
    def __init__(
        self,
        weight: float,
        device: torch.device,
        name: str = "CLoss",
    ):
        super().__init__(name=name, weight=weight, device=device)
        self.core_loss = torch.nn.NLLLoss(reduction='mean').to(device)

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pre_softmax = model_output.pre_softmax
        softmax_out = F.log_softmax(pre_softmax, dim=1)
        return self.core_loss(softmax_out, targets)


class TanhLoss(PartialLoss):
    def __init__(
        self,
        weight: float,
        device: torch.device,
        name: str = "TLoss",
        eps: float = 1e-10,
    ):
        super().__init__(name=name, weight=weight, device=device)
        self.eps = eps

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        fet_vec_1, fet_vec_2 = model_output.proto_feature_vec.chunk(2)
        t1_loss = self.core_loss(fet_vec_1)
        t2_loss = self.core_loss(fet_vec_2)
        return (t1_loss + t2_loss) / 2.0

    def core_loss(self, x: torch.Tensor) -> torch.Tensor:
        loss = torch.tanh(torch.sum(x, dim=0))
        loss = -torch.log(loss + self.eps).mean()
        return loss


class AlignLoss(PartialLoss):
    def __init__(
        self,
        weight: float,
        device: torch.device,
        name="ALoss",
        eps: float = 1e-10,
    ):
        super().__init__(name=name, weight=weight, device=device)
        self.eps = eps

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        feat_map_1, feat_map_2 = model_output.proto_feature_map.chunk(2)
        a1_loss = self.core_loss(feat_map_1, feat_map_2.detach())
        a2_loss = self.core_loss(feat_map_2, feat_map_1.detach())
        return (a1_loss + a2_loss) / 2.0

    def core_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        # from https://gitlab.com/mipl/carl/-/blob/main/losses.py
        z1 = z1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
        z2 = z2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
        loss = torch.einsum("nc,nc->n", [z1, z2])
        loss = -torch.log(loss + self.eps).mean()
        return loss

    def get_weight(self, epoch_idx: int, num_epochs: int) -> float:
        if self.weight is not None:
            return self.weight

        return epoch_idx / num_epochs


class AlignLossRotInv(PartialLoss):
    def __init__(
        self,
        weight: float,
        device: torch.device,
        name="ALoss",
        eps: float = 1e-10,
    ):
        super().__init__(name=name, weight=weight, device=device)
        self.eps = eps

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        feat_map_1, feat_map_2 = model_output.proto_feature_map.chunk(2)
        a1_loss = self.core_loss(feat_map_1, feat_map_2.detach(), loss_mask)
        a2_loss = self.core_loss(feat_map_2, feat_map_1.detach(), loss_mask)
        return (a1_loss + a2_loss) / 2.0

    def core_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Flatten H, W dims;
        N, D, _, _ = z1.shape
        z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
        z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)
        mask = loss_mask.permute(0, 2, 3, 1).reshape(N, -1)

        # Calculate inner product;
        x_inner = torch.sum(z1 * z2, dim=2)

        # Calculate masked loss;
        loss = -torch.log(x_inner + self.eps)
        loss *= mask
        loss = loss.sum() / mask.sum()
        return loss

    def get_weight(self, epoch_idx: int, num_epochs: int) -> float:
        if self.weight is not None:
            return self.weight

        return epoch_idx / num_epochs


class AlignLossRotMatch(PartialLoss):
    def __init__(
        self,
        weight: float,
        device: torch.device,
        name="ALoss",
        eps: float = 1e-10,
    ):
        super().__init__(
            name=name,
            weight=weight,
            device=device,
        )
        self.eps = eps

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        feat_map_1, feat_map_2 = model_output.proto_feature_map.chunk(2)
        a1_loss = self.core_loss(feat_map_1, feat_map_2.detach(), loss_mask)
        a2_loss = self.core_loss(feat_map_2, feat_map_1.detach(), loss_mask)
        return (a1_loss + a2_loss) / 2.0

    def core_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Flatten H, W dims;
        N, D, _, _ = z1.shape
        z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
        z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)

        # Calculate inner product;
        x_inner = torch.bmm(z1, z2.transpose(1, 2))

        # Calculate masked loss;
        loss = -torch.log(x_inner + self.eps)
        loss *= loss_mask
        loss = loss.sum() / loss_mask.sum()
        return loss

    def get_weight(self, epoch_idx: int, num_epochs: int) -> float:
        if self.weight is not None:
            return self.weight

        return epoch_idx / num_epochs


class WeightedSumLoss(torch.nn.Module):
    def __init__(
        self,
        partial_losses: List[PartialLoss],
        device: torch.device,
        num_epochs: int,
    ):
        super().__init__()
        # Partial losses for total loss computation;
        self.partial_losses = partial_losses
        self.device = device

        # Epoch tracking for adaptive weights;
        self.curr_epoch = 0
        self.num_epochs = num_epochs

        # Aggregation parameters;
        self.summed_val = 0.0
        self.num_steps = 0

    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Store partial loss values;
        loss_data = {}

        # Initialize total loss;
        total_loss = torch.tensor(0.0).to(self.device)

        # Calculate all partial losses;
        for loss_fn in self.partial_losses:
            result = loss_fn(
                model_output=model_output,
                targets=targets,
                loss_mask=loss_mask,
            )
            loss_fn_w = loss_fn.get_weight(
                epoch_idx=self.curr_epoch,
                num_epochs=self.num_epochs,
            )
            total_loss += loss_fn_w * result
            loss_data[loss_fn.name] = result

        loss_data["total_loss"] = total_loss

        # Aggregate result;
        with torch.no_grad:
            self.summed_val += total_loss
            self.num_steps += 1

        return loss_data

    def get_average_value(self) -> float:
        return self.summed_val / self.num_steps

    def reset(self):
        self.curr_epoch = 0
        self.num_epochs = 0
        self.summed_val = 0.0
        self.num_steps = 0

    def set_curr_epoch(self, epoch_idx: int):
        self.curr_epoch = epoch_idx


def get_align_loss(mode: str) -> Callable:
    """
    Selects training forward pass function;
    """
    if mode == "MATCH":
        return AlignLossRotMatch
    elif mode == "INV":
        return AlignLossRotInv
    elif mode == "PLAIN":
        return AlignLoss

    raise Exception(f"Training mode {mode} not implemented!")
