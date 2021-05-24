import argparse
import os
import time

from helpers.utils import (
    check_dirs_exist,
    get_device,
    accuracy,
    set_seeds,
    Logger
)
from helpers import data_loader
import models
from helpers.trainer import Trainer

from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description="Initial Train Process")
parser.add_argument('--n-epochs', default=200, type=int)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.1, 0.1, 0.1])
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{args.model}_{args.dataset}/initial_train/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = os.path.join(args.save_dir, "logs.txt")


class InitialModelTrainer(Trainer):
    def __init__(self, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_loss_and_backward(self, batch):
        input_var, target_var = batch
        output_var = self.model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        loss.backward()
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': loss.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, self.global_step
        )
        return loss, top1, top5

    def evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}


def main():
    set_seeds(args.seed)
    check_dirs_exist([args.save_dir])
    logger = Logger(args.log_path)
    if args.dataset not in data_loader.__dict__:
        raise NameError
    if args.model not in models.__dict__:
        raise NameError
    train_loader, eval_loader, num_classes = data_loader.__dict__[args.dataset](args.batch_size)
    model = models.__dict__[args.model](num_classes=num_classes)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    base_trainer_cfg = (args, model, train_loader, eval_loader, optimizer, args.save_dir, get_device(), logger)
    writer = SummaryWriter(log_dir=args.log_dir)  # For tensorboardX
    trainer = InitialModelTrainer(writer, *base_trainer_cfg)
    logger.log('\n'.join(map(str, vars(args).items())))
    trainer.train()
    print(f'Log Path : {args.log_path}')


if __name__ == '__main__':
    main()
