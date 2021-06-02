import argparse
import os
import time

from helpers.utils import (
    check_dirs_exist,
    accuracy,
    load_model,
    get_device,
    Logger
)
from helpers import dataset
from helpers.trainer import Trainer
from helpers.encoder import HuffmanEncoder
import models

import torch.nn as nn


parser = argparse.ArgumentParser(description="Encode Process")
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--load-model-path', type=str, default='None')
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{args.model}_{args.dataset}/encode/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = os.path.join(args.save_dir, "logs.txt")


class Evaluator(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _get_loss_and_backward(self, _):
        pass

    def _evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}


def main():
    check_dirs_exist([args.save_dir])
    logger = Logger(args.log_path)
    device = get_device()
    if args.dataset not in dataset.__dict__:
        raise NameError
    if args.model not in models.__dict__:
        raise NameError
    _, eval_loader, num_classes = dataset.__dict__[args.dataset](args.batch_size)
    enc_model = models.__dict__[args.model](num_classes=num_classes)
    load_model(enc_model, args.load_model_path, logger, device)
    base_cfg = (args, enc_model, _, eval_loader, _, args.save_dir, device, logger)
    evaluator = Evaluator(*base_cfg)
    evaluator.eval()
    encoder = HuffmanEncoder(logger)
    logger.log('\n'.join(map(str, vars(args).items())))

    # Encode
    encoder.huffman_encode_model(enc_model)

    # Decode
    dec_model = models.__dict__[args.model](num_classes=num_classes)
    encoder.huffman_decode_model(dec_model)
    base_cfg = (args, dec_model, _, eval_loader, _, args.save_dir, device, logger)
    evaluator = Evaluator(*base_cfg)
    evaluator.eval()

    print(f'Log Path : {args.log_path}')


if __name__ == '__main__':
    main()

