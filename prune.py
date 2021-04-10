import argparse
import os
import sys

from util import check_dirs_exist, get_device, accuracy, load_model, save_model
from data_loader import DataLoader
from models.alexnet import alexnet
from trainer import Trainer

from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description='Prune Process')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_drop', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--prune-mode', type=str, default='filter-ga')
parser.add_argument('--prune-rates', nargs='+', type=float, default=[0.16, 0.62, 0.65, 0.63, 0.63])
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # By default we will only prune once
parser.add_argument('--prune-retrain-epochs', type=int, default=200)
parser.add_argument('--prune-retrain-lr', type=float, default=0.0001)
parser.add_argument('--load-model-path', type=str)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{args.model}_{args.dataset}/prune'
args.log_dir = os.path.join(args.save_dir, 'log')


class PrunedModelTrainer(Trainer):
    def __init__(self, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.cross_entropy = nn.CrossEntropyLoss()
        self.filter_prune_rates = self.model.get_filters_prune_rates(self.args.prune_rates)
        self.last_epoch = None

    def get_loss(self, batch, global_step):
        input_var, target_var = batch

        # Do soft prune per "args.prune_interval"
        if self.last_epoch != self.cur_epoch and self.cur_epoch % self.args.prune_interval == 0:
            self.last_epoch = self.cur_epoch

            # In order to get the gradient and use it on the criterion "filter-ga"
            output_var = self.model(input_var)
            loss = self.cross_entropy(output_var, target_var)
            loss.backward(retain_graph=True)

            # Soft prune by the criterion "filter-ga"
            self.model.prune(args.prune_mode, args.prune_rates)

        # Evaluate
        output_var = self.model(input_var)
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
        print(f'\tLoss: {loss.item():.3f}\t'
              f'Top1: {top1:.3f}\t'
              f'Top5: {top5:.3f}')
        return loss.mean(), top1.mean(), top5.mean()

    def evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.model(input_var)
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
        load_model(model, args.load_model_path, get_device())
    else:
        raise ValueError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    base_trainer_cfg = (args, model, train_loader, eval_loader, optimizer, args.save_dir, get_device())
    writer = SummaryWriter(log_dir=args.log_dir)  # for tensorboardX
    trainer = PrunedModelTrainer(writer, *base_trainer_cfg)
    trainer.train()


if __name__ == '__main__':
    main()
