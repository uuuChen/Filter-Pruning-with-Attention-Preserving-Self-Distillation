import os
import random

import numpy as np
import torch


def set_seeds(seed):
    """ Set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """ Get device (CPU or GPU) """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def check_dirs_exist(path_or_path_list):
    if isinstance(path_or_path_list, str):
        os.makedirs(path_or_path_list, exist_ok=True)
    elif isinstance(path_or_path_list, list):
        for path in path_or_path_list:
            os.makedirs(path, exist_ok=True)
    else:
        raise AttributeError(path_or_path_list)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)  # pred shape: (batch_size, k)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = list()
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_model(model, file_path, device='cuda'):
    print('Loading the model from', file_path)
    if file_path is not None:
        model.load_state_dict(torch.load(file_path, map_location=device))


def save_model(model, file_path):
    print('Saving the model to', file_path)
    torch.save(model.state_dict(), file_path)


def get_average_meters(n=1):
    return [AverageMeter() for _ in range(n)]


class AverageMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, n=1):
        self.n += n
        self.sum += val * n
        self.mean = self.sum / self.n
