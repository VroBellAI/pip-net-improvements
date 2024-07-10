import os
import sys
import torch
from argparse import Namespace

from pipnet.pipnet import get_pipnet
from pipnet.initialization import (
    initialize_or_load_model,
)
from pipnet.loss import PIPNetLoss
from pipnet.optimizers import (
    get_backbone_optimizer,
    get_head_optimizer,
    get_backbone_scheduler,
    get_head_scheduler,
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
from util.logger import Logger, create_csv_log
from util.data import get_dataloaders
from util.args import get_args, save_args
from util.vis_pipnet import visualize_topk
from util.visualize_prediction import visualize_all


def run_pipnet(args: Namespace):
    # Create Logger;
    logger = Logger(args.log_dir)
    save_args(args, logger.metadata_dir)

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

    # Initialize optimizer & scheduler;
    backbone_optimizer = get_backbone_optimizer(
        optimizer_name=args.optimizer,
        network=pip_net,
        lr_network=args.lr,
        lr_backbone=args.lr_net,
        lr_block=args.lr_block,
        weight_decay=args.weight_decay,
    )

    backbone_scheduler = get_backbone_scheduler(
        backbone_optimizer=backbone_optimizer,
        num_epochs=args.epochs_pretrain,
        num_steps_per_epoch=len(loaders["pre_train"]),
        eta_min=args.lr_block / 100.0,
    )

    # Initialize or load model's params;
    initialize_or_load_model(
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        state_dict_dir_net=args.state_dict_dir_net,
        device=device,
    )

    # Define loss function;
    loss_func = PIPNetLoss(
        aug_mode=args.train_mode,
        class_norm_mul=pip_net.module.get_norm_mul(),
        device=device,
        eps=1e-7 if args.mixed_precision else 1e-10,  # <- numerical stability constant;
    )

    # Pass num prototypes to arguments;
    # TODO: remove in the later refactor
    args.wshape = pip_net.get_num_prototypes()
    print(f"Output shape: {args.wshape}", flush=True)

    # Create logging file;
    create_csv_log(logger, num_classes=num_classes)

    # Pretrain prototypes;
    pretrain_self_supervised(
        num_epochs=args.epochs_pretrain,
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        backbone_scheduler=backbone_scheduler,
        loss_func=loss_func,
        train_loader=loaders["pre_train"],
        logger=logger,
        state_dict_dir_net=args.state_dict_dir_net,
        device=device,
        use_mixed_precision=args.mixed_precision,
    )

    # Visualize top K prototypes;
    if 'convnext' in args.net and args.epochs_pretrain > 0:
        visualize_topk(
            net=pip_net,
            projectloader=loaders["project"],
            device=device,
            foldername='visualised_pretrained_prototypes_topk',
            args=args,
        )

    # Initialize optimizers and schedulers for second training phase;
    backbone_optimizer = get_backbone_optimizer(
        optimizer_name=args.optimizer,
        network=pip_net,
        lr_network=args.lr,
        lr_backbone=args.lr_net,
        lr_block=args.lr_block,
        weight_decay=args.weight_decay,
    )

    head_optimizer = get_head_optimizer(
        optimizer_name=args.optimizer,
        network=pip_net,
        lr_network=args.lr,
        weight_decay=args.weight_decay,
    )

    backbone_scheduler = get_backbone_scheduler(
        backbone_optimizer=backbone_optimizer,
        num_epochs=args.epochs,
        num_steps_per_epoch=len(loaders["train"]),
        eta_min=args.lr_net / 100.0,
    )

    head_scheduler = get_head_scheduler(
        head_optimizer=head_optimizer,
        num_epochs=args.epochs,
        eta_min=0.001,
    )

    # Train classifier;
    train_supervised(
        num_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        network=pip_net,
        backbone_optimizer=backbone_optimizer,
        backbone_scheduler=backbone_scheduler,
        head_optimizer=head_optimizer,
        head_scheduler=head_scheduler,
        loss_func=loss_func,
        train_loader=loaders["train"],
        test_loader=loaders["test"],
        logger=logger,
        state_dict_dir_net=args.state_dict_dir_net,
        device=device,
        epochs_to_finetune=3,
        pretrained=args.epochs_pretrain > 0,
        use_mixed_precision=args.mixed_precision,
    )

    # Visualize top K prototypes;
    top_ks = visualize_topk(
        net=pip_net,
        projectloader=loaders["project"],
        device=device,
        foldername='visualised_prototypes_topk',
        args=args,
    )

    # Zero out rarely-occurring prototypes;
    zero_idxs = pip_net.zero_out_irrelevant_protos(
        top_ks=top_ks,
        min_score=0.1,
    )
    print(
        f"Weights of prototypes {zero_idxs}"
        f"are set to zero because it is never detected "
        f"with similarity>0.1 in the training set",
        flush=True,
    )

    # Evaluate the model;
    eval_pipnet(
        net=pip_net,
        test_loader=loaders["test"],
        epoch=f"notused {args.epochs}",
        device=device,
        log=logger,
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
            log=logger,
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
        log=logger,
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
