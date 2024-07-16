import torch
import random
import numpy as np
from typing import List


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_patch_size(args):
    patch_size = 32
    skip = round((args.image_size - patch_size) / (args.wshape-1))
    return patch_size, skip


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(
            tensor=m.weight,
            gain=torch.nn.init.calculate_gain('sigmoid'),
        )


def connect_gradients(params: List[torch.nn.Parameter]):
    for param in params:
        param.requires_grad = True


def disconnect_gradients(params: List[torch.nn.Parameter]):
    for param in params:
        param.requires_grad = False
