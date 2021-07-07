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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:25} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) '
              f'| total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : '
          f'{total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')


def get_device(i=0):
    """ Get device (CPU or GPU) """
    device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")
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


def load_model(model, file_path, logger, device='cuda'):
    if file_path is not None:
        logger.log(f'Loading the model from {file_path}', verbose=True)
        model.load_state_dict(torch.load(file_path, map_location=device))


def save_model(model, file_path, logger):
    logger.log(f'Saving the model to {file_path}', verbose=True)
    torch.save(model.state_dict(), file_path)


def get_average_meters(n=1):
    return [AverageMeter() for _ in range(n)]


def z_score_v2(x):
    return (x - np.min(x)) / np.std(x)


def min_max_scalar(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, text, verbose=False):
        with open(self.log_path, "a") as f:
            f.write(f'{text}\n')
            f.flush()
            if verbose:
                print(text)

    def log_line(self):
        line = '\n' + '-' * 100
        self.log(line)


class AverageMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, n=1):
        self.n += n
        self.sum += val * n
        self.mean = self.sum / self.n
