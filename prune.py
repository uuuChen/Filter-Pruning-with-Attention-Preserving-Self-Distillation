import argparse
import os

from util import check_dirs_exist, get_device, accuracy
from data_loader import get_cifar100
from models.alexnet import alexnet
from trainer import Trainer

from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description='Prune Process')
parser.add_argument('--prune-mode', '-pm', default='filter-norm', type=str, metavar='M',
                    help='1. filter-norm 2. channel-norm 3. filter-gm\n')
parser.add_argument('--prune-rates', "-pr", nargs='+', type=float, default=[0.16, 0.62, 0.65, 0.63, 0.63],
                    help='pruning rate for AlexNet conv layer (default=[0.16, 0.62, 0.65, 0.63, 0.63])')
parser.add_argument('--prune-interval', '-pi', default=1, type=int,
                    metavar='N', help='prune interval when using prune-mode "filter-gm" (default: 1)')
parser.add_argument('--prune-retrain-epochs', '-prep', default=200, type=int,
                    metavar='N',  help='number of pruning retrain epochs to run (default: 200)')
parser.add_argument('--prune-retrain-lr', '-prlr', default=0.0001, type=float,
                    metavar='PRLR', help='pruning retrain learning rate')
parser.add_argument('--model-file',  default="saves/alexnet_cifar100/model_epochs_30.pt", type=str)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.log_dir = os.path.join(args.save_dir, 'log')


class PrunedModelTrainer(Trainer):
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
        print(f'\tLoss: {loss.item():.3f}\t'
              f'Top1: {top1:.3f}\t'
              f'Top5: {top5:.3f}')
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
        train_loader, eval_loader = get_cifar100(args.batch_size)  # get data loader
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
    trainer = PrunedModelTrainer(writer, *base_trainer_cfg)
    trainer.train(model_file=args.model_file)


if __name__ == '__main__':
    main()
