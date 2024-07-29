import torch
from pipnet.pipnet import PIPNet, PIPNetOutput

from typing import Dict, Union


class Metric(torch.nn.Module):
    def __init__(self, name: str, aggregation: str = "mean"):
        super().__init__()
        self.name = name
        self.aggregated_val = 0.0
        self.num_steps = 0
        self.aggregation = aggregation

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def get_aggregated_value(self) -> float:
        if self.aggregation == "mean":
            return self.aggregated_val / self.num_steps

        if self.aggregation == "sum":
            return self.aggregated_val

        if self.aggregation == "last":
            return self.aggregated_val

        raise Exception(f"Aggregation {self.aggregation} not implemented!")

    def reset(self):
        self.aggregated_val = 0.0
        self.num_steps = 0


class ClassAccuracy(Metric):
    """
    Basic class accuracy.
    """
    def __init__(self, name: str = "Acc", aggregation: str = "mean"):
        super().__init__(name=name, aggregation=aggregation)

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        # Calculate accuracy
        preds = torch.argmax(model_output.logits, dim=1)
        correct = torch.sum(torch.eq(preds, targets))
        result = correct / float(len(targets))

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += 1

        return result


class NumRelevantScores(Metric):
    """
    Calculates number of activation scores
    greater that the given score thresh.
    """
    def __init__(
        self,
        thresh: float = 0.1,
        name: str = "NumRelevantScores",
        aggregation: str = "mean",
    ):
        super().__init__(name=name, aggregation=aggregation)
        self.thresh = thresh

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Extract prototypes feature vector;
        proto_vec = model_output.proto_feature_vec

        # Calculate # relevant scores;
        relevant_scores = torch.relu(proto_vec - self.thresh)
        num_relevant_scores = torch.count_nonzero(relevant_scores, dim=1)
        result = num_relevant_scores.float().mean()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += 1

        return result


class NumAbstainedPredictions(Metric):
    """
    Calculates number of inputs
    for which model abstained from prediction.
    """
    def __init__(self, name: str = "NumAbstained", aggregation: str = "sum"):
        super().__init__(name=name, aggregation=aggregation)

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Extract model output;
        logits = model_output.logits
        batch_size = logits.shape[0]

        # Calculate non-zero class scores;
        logits_max_scores, _ = torch.max(logits, dim=1)
        num_relevant_predictions = torch.count_nonzero(logits_max_scores)
        result = (batch_size - num_relevant_predictions).float()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += 1

        return result


class ANZProto(Metric):
    """
    Calculates Almost Non-Zero sparsity estimate
    for prototype vector.
    """
    def __init__(
        self,
        threshold: float = 1e-3,
        name: str = "ANZProto",
        aggregation: str = "mean",
    ):
        super().__init__(name=name, aggregation=aggregation)
        self.threshold = threshold

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Extract batch_size for denominator;
        batch_size = targets.shape[0]

        # Calculate Almost Non-Zero activations;
        proto_vec = model_output.proto_feature_vec
        result = (torch.abs(proto_vec) > self.threshold).sum()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += batch_size

        return result / batch_size


class ANZSimScores(Metric):
    """
    Calculates Almost Non-Zero sparsity estimate
    for similarity scores between
    prototype feature vectors and predicted class representative vectors.
    """
    def __init__(
        self,
        network: PIPNet,
        threshold: float = 1e-3,
        name: str = "ANZSimScores",
        aggregation: str = "mean",
    ):
        super().__init__(name=name, aggregation=aggregation)
        self.threshold = threshold
        self.class_w = network.module.get_class_weight()

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Extract batch_size for denominator;
        batch_size = targets.shape[0]

        # Extract model output;
        proto_vec = model_output.proto_feature_vec
        logits = model_output.logits
        _, pred_classes = torch.max(logits, dim=1)

        # Extract weight vectors of predicted classes;
        pred_class_vec = torch.index_select(self.class_w, dim=0, index=pred_classes)

        # Calculate Almost Non-Zero similarities
        # Between proto-vectors and corresponding class vectors;
        sim_scores = proto_vec * pred_class_vec
        result = (torch.abs(sim_scores) > self.threshold).sum()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += batch_size

        return result / batch_size


class LocalSize(Metric):
    def __init__(
        self,
        network: PIPNet,
        threshold: float = 1e-3,
        name: str = "LocalSize",
        aggregation: str = "mean",
    ):
        super().__init__(name=name, aggregation=aggregation)
        self.threshold = threshold
        self.class_w = network.module.get_class_weight()

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        network: PIPNet,
        **kwargs,
    ) -> torch.Tensor:
        # Extract batch_size for denominator;
        batch_size = targets.shape[0]

        # Extract model output;
        proto_vec = model_output.proto_feature_vec

        # Extract classification weight matrix;
        class_w_repeat = self.class_w.unsqueeze(1).repeat(1, batch_size, 1)

        sim_scores = proto_vec * class_w_repeat
        sim_scores_thr = torch.relu(sim_scores - self.threshold).sum(dim=1)
        result = (sim_scores_thr > 0.0).sum()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += batch_size

        return result / batch_size


class NumNonZeroPrototypes(Metric):
    """
    Calculates number of prototypes
    with any non-zero weight in class matrix.
    """
    def __init__(
        self,
        network: PIPNet,
        threshold: float = 1e-3,
        name: str = "NumNonZeroPrototypes",
    ):
        super().__init__(name=name, aggregation="last")
        self.threshold = threshold
        self.class_w = network.module.get_class_weight()

    @torch.no_grad()
    def forward(self, **kwargs) -> torch.Tensor:
        result = (self.class_w > self.threshold).any(dim=0).sum()

        # Save last result as aggregated value;
        self.aggregated_val = result.detach().item()

        return result


class TopKClassAccuracy(Metric):
    def __init__(self, k: int, name: str = "", aggregation: str = "mean"):
        if name == "":
            name = f"Top{k}Acc"
        super().__init__(name=name, aggregation=aggregation)
        self.k = k

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Extract batch_size for denominator;
        batch_size = targets.shape[0]

        # Extract classification scores;
        logits = model_output.logits

        # Check for too big k;
        assert self.k <= logits.shape[1]

        # Calculate top K predicted class indices;
        _, top_k_indices = logits.topk(self.k, dim=1, largest=True, sorted=True)
        targets = targets.view(-1, 1)

        # Compare with targets;
        correct_predictions = torch.eq(top_k_indices, targets).sum(dim=1)

        # Aggregate comparison;
        result = correct_predictions.float().sum()

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += batch_size

        return result / batch_size


class NumInDistribution(Metric):
    """
    Calculates # In-Distribution Samples.
    """
    def __init__(
        self,
        threshold: Union[float, Dict],
        name: str = "InDistrSamples",
        aggregation: str = "sum",
    ):
        super().__init__(name=name, aggregation=aggregation)
        self.threshold = threshold

    @torch.no_grad()
    def forward(
        self,
        model_output: PIPNetOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        # Extract classification scores;
        logits = model_output.logits
        max_out_score, ys_pred = torch.max(logits, dim=1)

        if isinstance(self.threshold, float):
            result = (max_out_score >= self.threshold).sum()

        elif isinstance(self.threshold, dict):
            result = 0
            for smp_idx in range(len(ys_pred)):
                smp_thresh = self.threshold[ys_pred[smp_idx].item()]
                smp_score = logits[smp_idx, :]
                if smp_score.max().item() >= smp_thresh:
                    result += 1  # TODO: result as tensor!
        else:
            raise ValueError("provided threshold should be float or dict", type(self.threshold))

        # Aggregate result;
        self.aggregated_val += result.detach().item()
        self.num_steps += 1

        return result


# TODO: Sparsity ratio:
# print("sparsity ratio: ", (torch.numel(net.module._classification.weight)-torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight-1e-3)).item()) / torch.numel(net.module._classification.weight), flush=True)


def metric_data_to_str(metrics_data: Dict[str, torch.Tensor]) -> str:
    """
    Converts metrics data to string info.
    """
    metrics_info = ""

    for metric in metrics_data:
        metrics_info += f"{metric}: {metrics_data[metric]:.3f} | "

    return metrics_info


