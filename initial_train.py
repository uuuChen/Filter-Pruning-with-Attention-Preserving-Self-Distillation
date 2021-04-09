import argparse
import os

from util import check_dirs_exist, get_device, accuracy, load_model
from data_loader import DataLoader
from models.alexnet import alexnet
from trainer import Trainer

from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description="Initial Train Process")
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_drop', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--save_dir', type=str, default='saves/default')
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.log_dir = os.path.join(args.save_dir, 'log')


class InitialModelTrainer(Trainer):
    def __init__(self, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_loss(self, model, batch, global_step):
        input_var, target_var = batch
        output_var = model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': loss.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, global_step
        )
        return loss.mean(), top1.mean(), top5.mean()

    def evaluate(self, model, batch):
        input_var, target_var = batch
        output_var = model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}


def main():
    check_dirs_exist([args.save_dir])
    if args.dataset == 'cifar100':
        train_loader, eval_loader = DataLoader.get_cifar100(args.batch_size)  # get data loader
        num_classes = 100
    else:
        raise ValueError
    if args.model == 'alexnet':
        model = alexnet(num_classes=num_classes)
    else:
        raise ValueError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    base_trainer_cfg = (args, model, train_loader, eval_loader, optimizer, args.save_dir, get_device())
    writer = SummaryWriter(log_dir=args.log_dir)  # for tensorboardX
    trainer = InitialModelTrainer(writer, *base_trainer_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
