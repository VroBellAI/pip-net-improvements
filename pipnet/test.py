import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
from util.logger import Logger
from util.data import get_dataloaders
from util.eval_cub_csv import (
    eval_prototypes_cub_parts_csv,
    get_topk_cub,
    get_proto_patches_cub,
)
from argparse import Namespace
from copy import deepcopy
from typing import List, Dict, Union

from pipnet.pipnet import PIPNet
from pipnet.metrics import (
    Metric,
    NumAbstainedPredictions,
    NumInDistribution,
    metric_data_to_str,
)


# TODO: LOGGER calls!!!
@torch.no_grad()
def test_step(
    network: PIPNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[Metric],
) -> Dict[str, torch.Tensor]:

    # Predict inputs;
    model_output = network(inputs, inference=True)

    metrics_data = {
        metric.name: metric(
            model_output=model_output,
            targets=targets,
            network=network,
        )
        for metric in metrics
    }
    return metrics_data


@torch.no_grad()
def evaluate_pipnet(
    network: PIPNet,
    test_loader: DataLoader,
    metrics: List[Metric],
    epoch_idx: int,
    device: torch.device,
    phase: str = "evaluation"
) -> Dict[str, float]:
    
    # Make sure the model is in evaluation mode;
    network.eval()

    # Reset metrics aggregation;
    [metric.reset() for metric in metrics]

    # Show progress on progress bar
    test_iter = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"{phase} ep{epoch_idx}",
        mininterval=5.0,
        ncols=0,
    )

    # Evaluation loop;
    for step_idx, (x, y) in test_iter:
        x, y = x.to(device), y.to(device)

        step_info = test_step(
            network=network,
            inputs=x,
            targets=y,
            metrics=metrics,
        )

        # Print metrics data;
        test_iter.set_postfix_str(
            s=metric_data_to_str(step_info),
            refresh=False,
        )

    # Save aggregated metrics;
    test_info = {}

    for metric in metrics:
        test_info[f"test_{metric.name}"] = metric.get_aggregated_value()

    return test_info


@torch.no_grad()
# Calculates class-specific threshold for the FPR@X metric. Also calculates threshold for images with correct prediction (currently not used, but can be insightful)
def get_thresholds(net,
                   test_loader: DataLoader,
                   epoch,
                   device,
                   percentile:float = 95.,
                   log: Logger = None,
                   log_prefix: str = 'log_eval_epochs',
                   progress_prefix: str = 'Get Thresholds Epoch'
                   ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
    
    outputs_per_class = dict()
    outputs_per_correct_class = dict()
    for c in range(net.module._num_classes):
        outputs_per_class[c] = []
        outputs_per_correct_class[c] = []
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s Perc %s'%(epoch,percentile),
                        mininterval=5.,
                        ncols=0)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs)

            ys_pred = torch.argmax(out, dim=1)
            for pred in range(len(ys_pred)):
                outputs_per_class[ys_pred[pred].item()].append(out[pred,:].max().item())
                if ys_pred[pred].item()==ys[pred].item():
                    outputs_per_correct_class[ys_pred[pred].item()].append(out[pred,:].max().item())
        
        del out
        del pooled
        del ys_pred

    class_thresholds = dict()
    correct_class_thresholds = dict()
    all_outputs = []
    all_correct_outputs = []
    for c in range(net.module._num_classes):
        if len(outputs_per_class[c])>0:
            outputs_c = outputs_per_class[c]
            all_outputs += outputs_c
            class_thresholds[c] = np.percentile(outputs_c,100-percentile) 
            
        if len(outputs_per_correct_class[c])>0:
            correct_outputs_c = outputs_per_correct_class[c]
            all_correct_outputs += correct_outputs_c
            correct_class_thresholds[c] = np.percentile(correct_outputs_c,100-percentile)
    
    overall_threshold = np.percentile(all_outputs,100-percentile)
    overall_correct_threshold = np.percentile(all_correct_outputs,100-percentile)
    # if class is not predicted there is no threshold. we set it as the minimum value for any other class 
    mean_ct = np.mean(list(class_thresholds.values()))
    mean_cct = np.mean(list(correct_class_thresholds.values()))
    for c in range(net.module._num_classes):
        if c not in class_thresholds.keys():
            print(c,"not in class thresholds. Setting to mean threshold", flush=True)
            class_thresholds[c] = mean_ct
        if c not in correct_class_thresholds.keys():
            correct_class_thresholds[c] = mean_cct

    calculated_percentile = 0
    correctly_classified = 0
    total = 0
    for c in range(net.module._num_classes):
        correctly_classified+=sum(i>class_thresholds[c] for i in outputs_per_class[c])
        total += len(outputs_per_class[c])
    calculated_percentile = correctly_classified/total

    if percentile<100:
        while calculated_percentile < (percentile/100.):
            class_thresholds.update((x, y*0.999) for x, y in class_thresholds.items())
            correctly_classified = 0
            for c in range(net.module._num_classes):
                correctly_classified+=sum(i>=class_thresholds[c] for i in outputs_per_class[c])
            calculated_percentile = correctly_classified/total

    return overall_correct_threshold, overall_threshold, correct_class_thresholds, class_thresholds


def get_in_distribution_fraction(
    network: PIPNet,
    test_loader: DataLoader,
    epoch: int,
    device: torch.device,
    threshold: Union[float, Dict],
) -> float:

    abstained_metric = NumAbstainedPredictions(aggregation="sum")
    id_metric = NumInDistribution(threshold=threshold, aggregation="sum")

    evaluate_pipnet(
        network=network,
        test_loader=test_loader,
        metrics=[abstained_metric, id_metric],
        epoch_idx=epoch,
        device=device,
        phase="ood_evaluation"
    )

    num_samples = len(test_loader) * test_loader.batch_size
    num_abstained = abstained_metric.get_aggregated_value().item()
    num_id_samples = id_metric.get_aggregated_value().item()

    print(
        f"Samples seen: {num_samples}, "
        f"of which predicted as In-Distribution: {num_id_samples}",
        flush=True,
    )
    print(
        f"PIP-Net abstained from a decision for {num_abstained} images",
        flush=True,
    )
    return num_id_samples/num_samples


def evaluate_ood_detection(
    epoch: int,
    network: PIPNet,
    test_data_loader: DataLoader,
    ood_datasets: List[str],
    percentile: float,
    log: Logger,
    args: Namespace,
    device: torch.device,
):

    print(
        f"OOD Evaluation for epoch {epoch}, "
        f"with percent of {percentile}",
        flush=True,
    )
    _, _, _, class_thresholds = get_thresholds(
        net=network,
        test_loader=test_data_loader,
        epoch=epoch,
        device=device,
        percentile=percentile,
        log=log,
    )
    print(f"Thresholds: {class_thresholds}", flush=True)

    # Evaluate with in-distribution data;
    id_fraction = get_in_distribution_fraction(
        network=network,
        test_loader=test_data_loader,
        epoch=epoch,
        device=device,
        threshold=class_thresholds,
    )
    print(
        f"ID class threshold ID fraction (TPR) "
        f"with percent {percentile}: {id_fraction}",
        flush=True,
    )

    # Evaluate with out-of-distribution data;
    for ood_dataset in ood_datasets:
        # Omit used dataset;
        if ood_dataset == args.dataset:
            continue

        print(f"OOD dataset: {ood_dataset}", flush=True)
        ood_args = deepcopy(args)
        ood_args.dataset = ood_dataset
        loaders, _ = get_dataloaders(ood_args, device)

        id_fraction = get_in_distribution_fraction(
            network=network,
            test_loader=loaders["test"],
            epoch=epoch,
            device=device,
            threshold=class_thresholds,
        )
        print(
            f"{args.dataset} - OOD {ood_dataset} class threshold ID fraction (FPR) "
            f"with percent {percentile}: {id_fraction}",
            flush=True,
        )


def display_proto_data(
    network: PIPNet,
    data_loader: DataLoader,
    validation_size: float,
):

    head_mtrx = network.module.get_class_weight()
    head_mtrx_non_zero = head_mtrx[head_mtrx.nonzero(as_tuple=True)]
    head_bias = network.module.get_class_bias()

    # Print prototypes data;
    print(
        f"Classifier weights:"
        f"{head_mtrx}"
        f"{head_mtrx.shape}",
        flush=True,
    )
    print(
        f"Classifier weights nonzero:"
        f"{head_mtrx_non_zero}"
        f"{head_mtrx_non_zero.shape}",
        flush=True,
    )

    if head_bias is not None:
        print(
            f"Classifier bias:"
            f"{head_bias}"
            f"{head_bias.shape}",
            flush=True,
        )

    # Print weights and relevant prototypes per class;
    class_names = list(data_loader.dataset.class_to_idx.keys())
    class_idxs = list(data_loader.dataset.class_to_idx.values())

    num_classes, num_prototypes = head_mtrx.shape
    for c_idx in range(num_classes):
        relevant_ps = []
        proto_weights = head_mtrx[c_idx, :]

        for p_idx in range(num_prototypes):
            if proto_weights[p_idx] > 1e-3:
                relevant_ps.append((p_idx, proto_weights[p_idx].item()))

        if validation_size != 0.0:
            continue

        print(
            f"Class {c_idx}"
            f"({class_names[class_idxs.index(c_idx)]}):"
            f"has {len(relevant_ps)} relevant prototypes: {relevant_ps}",
            flush=True,
        )


def evaluate_prototype_purity(
    epoch: int,
    network: torch.nn.Module,
    train_data_loader,
    test_data_loader,
    args: Namespace,
    log: Logger,
    device: torch.device,
    threshold: float = 0.5,
):
    projectset_img0_path = train_data_loader.dataset.samples[0][0]
    project_path = os.path.split(os.path.split(projectset_img0_path)[0])[0].split("dataset")[0]
    parts_loc_path = os.path.join(project_path, "parts/part_locs.txt")
    parts_name_path = os.path.join(project_path, "parts/parts.txt")
    imgs_id_path = os.path.join(project_path, "images.txt")

    network.eval()
    # Evaluate for train set;
    print("\n\nEvaluating cub prototypes for training set", flush=True)
    csvfile_topk = get_topk_cub(
        net=network,
        projectloader=train_data_loader,
        k=10,
        epoch=f'train_{epoch}',
        device=device,
        args=args,
    )
    eval_prototypes_cub_parts_csv(
        csvfile=csvfile_topk,
        parts_loc_path=parts_loc_path,
        parts_name_path=parts_name_path,
        imgs_id_path=imgs_id_path,
        epoch=f'train_topk_{epoch}',
        args=args,
        log=log,
    )

    csvfile_all = get_proto_patches_cub(
        net=network,
        projectloader=train_data_loader,
        epoch=f'train_all_{epoch}',
        device=device,
        args=args,
        threshold=threshold,
    )
    eval_prototypes_cub_parts_csv(
        csvfile=csvfile_all,
        parts_loc_path=parts_loc_path,
        parts_name_path=parts_name_path,
        imgs_id_path=imgs_id_path,
        epoch=f'train_all_thres{threshold}_{epoch}',
        args=args,
        log=log,
    )

    # Evaluate for test set;
    print("\n\nEvaluating cub prototypes for test set", flush=True)
    csvfile_topk = get_topk_cub(
        net=network,
        projectloader=test_data_loader,
        k=10,
        epoch=f'test_{epoch}',
        device=device,
        args=args,
    )
    eval_prototypes_cub_parts_csv(
        csvfile=csvfile_topk,
        parts_loc_path=parts_loc_path,
        parts_name_path=parts_name_path,
        imgs_id_path=imgs_id_path,
        epoch=f'test_topk_{epoch}',
        args=args,
        log=log,
    )

    csvfile_all = get_proto_patches_cub(
        net=network,
        projectloader=test_data_loader,
        epoch=f'test_{epoch}',
        device=device,
        args=args,
        threshold=threshold,
    )
    eval_prototypes_cub_parts_csv(
        csvfile=csvfile_all,
        parts_loc_path=parts_loc_path,
        parts_name_path=parts_name_path,
        imgs_id_path=imgs_id_path,
        epoch=f'test_all_thres{threshold}_{epoch}',
        args=args,
        log=log,
    )