import torch
import argparse
from typing import Tuple, List


def get_device(args: argparse.Namespace) -> Tuple[torch.device, List[int]]:

    device_ids = []
    if args.gpu_ids != '':
        gpu_list = args.gpu_ids.split(',')
        device_ids = [int(gpu_id) for gpu_id in gpu_list]

    if args.disable_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"Device used: {device}, with id {device_ids}", flush=True)
        return device, device_ids

    num_devices = len(device_ids)

    if num_devices == 0:
        device = torch.device('cuda')
        device_ids.append(torch.cuda.current_device())
        print("CUDA device set without id specification", flush=True)
        print(f"Device used: {device}, with id {device_ids}", flush=True)
        return device, device_ids

    if num_devices == 1:
        device = torch.device(f"cuda: {args.gpu_ids}")
        print(f"Device used: {device}, with id {device_ids}", flush=True)
        return device, device_ids

    print(
        "This code should work with multiple GPU's "
        "but we didn't test that, "
        "so we recommend to use only 1 GPU.",
        flush=True,
    )
    device = torch.device('cuda:' + str(device_ids[0]))
    print(f"Device used: {device}, with id {device_ids}", flush=True)
    return device, device_ids
