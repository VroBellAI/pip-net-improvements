import os
import sys
import torch
from argparse import Namespace

from pipnet.pipnet import get_pipnet
from pipnet.initialization import (
    initialize_or_load_model,
    get_feature_vec_dim,
    zero_out_irrelevant_protos,
)
from pipnet.train_phases import (
    pretrain_self_supervised,
    train_supervised,
)
from pipnet.test import (
    eval_pipnet,
    evaluate_prototype_purity,
    evaluate_ood_detection,
    display_proto_data,
)
from util.device import get_device
from util.func import set_random_seed
from util.log import Log, create_csv_log
from util.data import get_dataloaders
from util.args import get_args, save_args, get_optimizer_nn
from util.vis_pipnet import visualize_topk
from util.visualize_prediction import visualize_all


def run_pipnet(args: Namespace):
    # Create Logger;
    log = Log(args.log_dir)
    save_args(args, log.metadata_dir)
    print(f"Log dir: {args.log_dir}", flush=True)

    # Select device;
    device, device_ids = get_device(args)

    # Create data loaders;
    loaders, classes = get_dataloaders(args, device)
    num_classes = len(classes)

    # Create PIP-Net;
    pip_net = get_pipnet(
        num_classes=num_classes,
        num_features=args.num_features,
        backbone_name=args.net,
        pretrained=not args.disable_pretrained,
        bias=args.bias,
    )
    pip_net = pip_net.to(device=device)
    pip_net = torch.nn.DataParallel(pip_net, device_ids=device_ids)

    # Create optimizers (w. schedulers);
    optimizers, params = get_optimizer_nn(pip_net, args)
    backbone_optimizer = optimizers["backbone"]
    head_optimizer = optimizers["head"]

    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        backbone_optimizer,
        T_max=len(loaders["pre_train"]) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # Initialize or load model's params;
    initialize_or_load_model(
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        head_optimizer=head_optimizer,
        num_classes=num_classes,
        num_prototypes=pip_net.module._num_prototypes,
        state_dict_dir_net=args.state_dict_dir_net,
        bias=args.bias,
        device=device,
    )

    # Define classification loss function;
    criterion = torch.nn.NLLLoss(reduction='mean').to(device)

    # Get the latent output size;
    feature_vec_shape = get_feature_vec_dim(
        network=pip_net,
        train_loader=loaders["train"],
        device=device,
    )
    args.wshape = feature_vec_shape[-1]
    print("Output shape: ", feature_vec_shape, flush=True)

    # Create logging file;
    create_csv_log(log, num_classes=pip_net.module._num_classes)

    # Pretrain prototypes;
    pretrain_self_supervised(
        num_epochs=args.epochs_pretrain,
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        backbone_scheduler=backbone_scheduler,
        head_optimizer=head_optimizer,
        criterion=criterion,
        train_loader=loaders["pre_train"],
        log=log,
        log_dir=args.log_dir,
        train_mode=args.train_mode,
        state_dict_dir_net=args.state_dict_dir_net,
        params_to_train=params["train"],
        params_to_freeze=params["freeze"],
        params_backbone=params["backbone"],
        device=device,
    )

    # Visualize top K prototypes;
    if 'convnext' in args.net and args.epochs_pretrain > 0:
        visualize_topk(
            net=pip_net,
            projectloader=loaders["project"],
            num_classes=num_classes,
            device=device,
            foldername='visualised_pretrained_prototypes_topk',
            args=args,
        )

    # Train classifier;
    # Re-initialize optimizers and schedulers for second training phase;
    optimizers, params = get_optimizer_nn(pip_net, args)
    backbone_optimizer = optimizers["backbone"]
    head_optimizer = optimizers["head"]

    backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        backbone_optimizer,
        T_max=len(loaders["train"])*args.epochs,
        eta_min=args.lr_net/100.0,
    )

    # Scheduler for the classification layer is with restarts,
    # Such that the model can re-active zeroed-out prototypes.
    # Hence an intuitive choice.
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        head_optimizer,
        T_0=5 if args.epochs <= 30 else 10,
        eta_min=0.001,
        T_mult=1,
        verbose=False,
    )

    # Train classifier;
    train_supervised(
        num_epochs=args.epochs,
        pretrain_epochs=args.epochs_pretrain,
        freeze_epochs=args.freeze_epochs,
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        backbone_scheduler=backbone_scheduler,
        head_optimizer=head_optimizer,
        head_scheduler=head_scheduler,
        criterion=criterion,
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        bias=args.bias,
        log=log,
        log_dir=args.log_dir,
        train_mode=args.train_mode,
        state_dict_dir_net=args.state_dict_dir_net,
        params_to_train=params["train"],
        params_to_freeze=params["freeze"],
        params_backbone=params["backbone"],
        device=device,
    )

    # Visualize top K prototypes;
    top_ks = visualize_topk(
        net=pip_net,
        projectloader=loaders["project"],
        num_classes=num_classes,
        device=device,
        foldername='visualised_prototypes_topk',
        args=args,
    )

    zero_out_irrelevant_protos(
        top_ks=top_ks,
        network=pip_net,
        min_score=0.1,
    )

    eval_pipnet(
        net=pip_net,
        test_loader=loaders["test"],
        epoch=f"notused {args.epochs}",
        device=device,
        log=log,
    )

    # Display prototypes info;
    display_proto_data(
        network=pip_net,
        data_loader=loaders["test"],
        validation_size=args.validation_size,
    )

    # Evaluate prototype purity;
    if args.dataset == 'CUB-200-2011':
        evaluate_prototype_purity(
            epoch=args.epochs,
            network=pip_net,
            train_data_loader=loaders["project"],
            test_data_loader=loaders["test_project"],
            args=args,
            log=log,
            device=device,
            threshold=0.5,
        )

    visualize_all(
        network=pip_net,
        proj_loader=loaders["project"],
        test_proj_loader=loaders["test_project"],
        classes=classes,
        device=device,
        args=args,
    )

    # Evaluate out of distribution detection;
    evaluate_ood_detection(
        epoch=args.epochs,
        network=pip_net,
        test_data_loader=loaders["test"],
        ood_datasets=["CARS", "CUB-200-2011", "pets"],
        percentile=95.0,
        log=log,
        args=args,
        device=device
    )

    print("Done!", flush=True)


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    # Set output filepaths;
    print_dir = os.path.join(args.log_dir, 'out.txt')
    tqdm_dir = os.path.join(args.log_dir, 'tqdm.txt')

    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()

    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')

    # Run training;
    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()
