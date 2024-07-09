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


# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=3662215#gistcomment-3662215
def topk_accuracy(output, target, topk=[1,]):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        topk2 = [x for x in topk if x <= output.shape[1]] #ensures that k is not larger than number of classes
        maxk = max(topk2)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            if k in topk2:
                correct_k = correct[:k].reshape(-1).float()
                res.append(correct_k)
            else:
                res.append(torch.zeros_like(target))
        return res