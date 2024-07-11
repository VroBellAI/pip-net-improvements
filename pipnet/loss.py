import torch
import torch.nn.functional as F
from typing import Dict, Optional

LOSS_DATA = Dict[str, torch.Tensor]


class PIPNetLoss(torch.nn.Module):
    def __init__(
        self,
        aug_mode: str,
        class_norm_mul: float,
        device: torch.device,
        print_info: bool = True,
        relevant_score_thresh: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.aug_mode = aug_mode
        self.device = device
        self.print_info = print_info
        self.relevant_score_thresh = relevant_score_thresh
        self.eps = eps

        # Track training mode and status;
        self.pretrain = False
        self.finetune = False
        self.num_epochs = None
        self.curr_epoch = None

        # Initialize loss weights;
        self.aw = None
        self.tw = None
        self.cw = None
        self.uw = None

        # Set classification loss params;
        self.class_norm_mul = class_norm_mul
        self.class_criterion = torch.nn.NLLLoss(reduction='mean').to(device)

    def set_pretrain_mode(self):
        """
        Sets loss weights to pre-train mode.
        """
        self.pretrain = True
        self.finetune = False
        # Set alignment loss weight later,
        # dynamically during training;
        self.aw = None
        self.tw = 5.0
        self.cw = 0.0
        self.uw = 0.5

    def set_train_mode(self):
        """
        Sets loss weights to train mode.
        """
        self.pretrain = False
        self.aw = 5.0
        self.tw = 2.0
        self.cw = 2.0
        self.uw = 0.0

    def set_num_epochs(self, num_epochs: int):
        """
        Sets num_epochs for tracking.
        """
        self.num_epochs = num_epochs

    def set_curr_epoch(self, epoch_idx: int):
        """
        Sets current epoch index for tracking.
        """
        self.curr_epoch = epoch_idx

    def forward(
        self,
        proto_features: torch.Tensor,
        pooled: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
    ) -> LOSS_DATA:

        # Store partial loss values;
        loss_data = {}

        # Extract tensor batches;
        targets = torch.cat([targets, targets])
        pooled1, pooled2 = pooled.chunk(2)
        pf1, pf2 = proto_features.chunk(2)

        # Initialize total loss;
        total_loss = torch.tensor(0.0).to(self.device)

        # Calculate auxiliary losses;
        a1_loss = self.align_loss(pf1, pf2.detach(), loss_mask)
        a2_loss = self.align_loss(pf2, pf1.detach(), loss_mask)
        a_loss = (a1_loss + a2_loss) / 2.0
        t_loss = (self.tanh_loss(pooled1) + self.tanh_loss(pooled2)) / 2.0

        loss_data["a_loss"] = a_loss
        loss_data["t_loss"] = t_loss

        # If pre-training,
        # Calculate alignment loss weight based on epoch idx;
        if self.pretrain:
            self.aw = self.curr_epoch / self.num_epochs

        if not self.finetune:
            total_loss += self.aw * a_loss + self.tw * t_loss

        # If not pre-training,
        # Calculate class loss and accuracy;
        if not self.pretrain:
            c_loss = self.class_loss(
                logits=logits,
                targets=targets,
            )
            total_loss += self.cw * c_loss
            acc = self.class_accuracy(logits, targets)

            loss_data["c_loss"] = c_loss
            loss_data["acc"] = acc

        # Add number of relevant scores to loss data;
        loss_data["total_loss"] = total_loss
        loss_data["num_relevant_scores"] = self.get_num_relevant_scores(pooled)
        return loss_data

    def align_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates alignment loss.
        Considers different differentiable affine
        data augmentation scenarios.
        """
        assert z1.shape == z2.shape
        assert z2.requires_grad is False

        if self.aug_mode == "MATCH":
            # Flatten H, W dims;
            N, D, _, _ = z1.shape
            z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
            z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)

            # Calculate inner product;
            x_inner = torch.bmm(z1, z2.transpose(1, 2))

            # Calculate masked loss;
            loss = -torch.log(x_inner + self.eps)
            loss *= mask
            loss = loss.sum() / mask.sum()
            return loss

        if self.aug_mode == "INV":
            # Flatten H, W dims;
            N, D, _, _ = z1.shape
            z1 = z1.permute(0, 2, 3, 1).reshape(N, -1, D)
            z2 = z2.permute(0, 2, 3, 1).reshape(N, -1, D)
            mask = mask.permute(0, 2, 3, 1).reshape(N, -1)

            # Calculate inner product;
            x_inner = torch.sum(z1 * z2, dim=2)

            # Calculate masked loss;
            loss = -torch.log(x_inner + self.eps)
            loss *= mask
            loss = loss.sum() / mask.sum()
            return loss

        if self.aug_mode == "PLAIN":
            # from https://gitlab.com/mipl/carl/-/blob/main/losses.py
            z1 = z1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
            z2 = z2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
            loss = torch.einsum("nc,nc->n", [z1, z2])
            loss = -torch.log(loss + self.eps).mean()
            return loss

        raise Exception(f"Augmentation mode {self.aug_mode} not implemented!")

    def tanh_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        loss = torch.tanh(torch.sum(inputs, dim=0))
        loss = -torch.log(loss + self.eps).mean()
        return loss

    def class_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classification loss with softmax and pre-softmax norm.
        """
        logits = torch.log1p(logits ** self.class_norm_mul)
        y_pred = F.log_softmax(logits, dim=1)
        return self.criterion(y_pred, targets)

    def uniform_loss(self, x: torch.Tensor, t=2) -> torch.Tensor:
        """
        Extra uniform loss from:
        https://www.tongzhouwang.info/hypersphere/
        Currently not used, but you could try adding it if you want.
        """
        loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + self.eps).log()
        return loss

    @staticmethod
    def class_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Basic class accuracy."""
        preds = torch.argmax(logits, dim=1)
        correct = torch.sum(torch.eq(preds, targets))
        return correct / float(len(targets))

    @torch.no_grad()
    def get_num_relevant_scores(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Calculates number of activation scores
        greater that self.relevant_score_thresh.
        """
        relevant_scores = torch.relu(pooled - self.relevant_score_thresh)
        return torch.count_nonzero(relevant_scores, dim=1).float().mean()

    def loss_data_to_str(self, loss_data: LOSS_DATA) -> str:
        """
        Converts loss data to string info.
        """
        loss_str = ""

        if "total_loss" in loss_data:
            loss_str += f"L:{loss_data['total_loss'].item():.3f} "

        if "c_loss" in loss_data:
            loss_str += f"LC:{loss_data['c_loss'].item():.3f} "

        if "a_loss" in loss_data:
            loss_str += f"LA:{loss_data['a_loss'].item():.2f} "

        if "t_loss" in loss_data:
            loss_str += f"LT:{loss_data['t_loss'].item():.3f} "

        if "num_relevant_scores" in loss_data:
            loss_str += (f"num_scores>{self.relevant_score_thresh}:"
                         f"{loss_data['num_relevant_scores'].item():.1f} ")

        if "acc" in loss_data:
            loss_str += f"Ac:{loss_data['acc'].item():.3f}"

        return loss_str
