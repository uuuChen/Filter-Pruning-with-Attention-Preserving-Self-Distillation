import argparse
import os
import sys
import time

from helpers.utils import (
    check_dirs_exist,
    get_device,
    accuracy,
    load_model,
    print_nonzeros,
    set_seeds,
    Logger
)
from helpers import dataset
import models
from helpers.trainer import Trainer
from helpers.pruner import FiltersPruner
from distillers_zoo import (
    MultiSimilarity,
    KLDistiller,
    Similarity,
    Attention,
    MultiAttention,
)

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn


parser = argparse.ArgumentParser(description='Attention Distilled With Pruned Model Process')
parser.add_argument('--n-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='resnet56')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160])
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.2, 0.2, 0.2])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--prune-mode', type=str, default='None')
parser.add_argument('--soft-prune', action='store_true', default=False)  # Do soft pruning or not
parser.add_argument('--prune-rates', nargs='+', type=float, default=[1.0])  # No prune by default
parser.add_argument('--samp-batches', type=int, default=None)  # Sample batches to compute gradient for pruning. Use
# all batches by default
parser.add_argument('--use-actPR', action='store_true', default=False)  # Compute actual pruning rates for conv layers
# or not
parser.add_argument('--use-greedy', action='store_true', default=False)  # Prune filters by greedy or independent
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # Do pruning process once by default
parser.add_argument('--distill', type=str, default='None')  # Which distillation methods to use
parser.add_argument('--window-size', type=int, default=None)  # Window size for "MAT" distillation. Determine how
# many layers of teacher are going to distill to all layers of students. Use all layers of teacher by default
parser.add_argument('--kd-T', type=float, default=4.0)  # Temperature for KL distillation
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=50.0)  # For custom-method distillation
parser.add_argument('--t-load-model-path', type=str, default='None')
parser.add_argument('--s-load-model-path', type=str, default='None')
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = f'saves/logs.txt'
if args.t_load_model_path is 'None':
    args.t_load_model_path = args.s_load_model_path


class PMSPModelTrainer(Trainer):
    """  A trainer for gradually self-distillation combined with attention mechanism and hard or soft pruning. """
    def __init__(self, t_model, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_model = t_model
        self.s_model = self.model
        self.writer = writer

        self.do_prune = self.args.prune_mode is not 'None'
        self.do_dist = self.args.distill is not 'None'
        self.do_soft_prune = self.args.soft_prune
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_div = KLDistiller(self.args.kd_T)
        if self.do_dist:
            self.criterion_kd, self.is_group, self.is_block = self._init_kd(self.args.distill)

        self.s_model_pruner = FiltersPruner(
            self.s_model,
            self.optimizer,
            self.train_loader,
            self.logger,
            samp_batches=self.args.samp_batches,
            device=self.device,
            use_actPR=self.args.use_actPR,
            use_greedy=self.args.use_greedy
        )
        self.last_epoch = None

        self.t_model.eval()
        self.t_model = self.t_model.to(self.device)

    def _mask_prune_weight_grad(self):
        conv_mask = self.s_model_pruner.get_conv_mask()
        for name, module in self.s_model.named_modules():
            if name in conv_mask:
                grad = module.weight.grad
                grad.data = grad.data * conv_mask[name]

    def _init_kd(self, method):
        is_group = False
        is_block = False
        if method == 'msp':
            is_block = True
            criterion = MultiSimilarity()
        elif method == 'sp':
            is_group = True
            criterion = Similarity()
        elif method == 'mat':
            is_block = True
            criterion = MultiAttention(window_size=self.args.window_size)
        elif method == 'at':
            is_group = True
            criterion = Attention()
        else:
            raise NotImplementedError(method)
        return criterion, is_group, is_block

    def _get_dist_feat(self, method, s_feat, t_feat):
        if method == 'msp':
            s_f = s_feat
            t_f = t_feat[-3:]
        elif method == 'mat':
            s_f = s_feat[1:-1]
            t_f = t_feat[1:-1]
        elif method == 'at':
            s_f = s_feat[1:-1]  # Get features g1 ~ g3
            t_f = s_feat[1:-1]  # Get features g1 ~ g3
        elif method == 'sp':
            s_f = [s_feat[-2]]  # Get g3 only
            t_f = [s_feat[-2]]  # Get g3 only
        else:
            raise NotImplementedError(method)
        return s_f, t_f

    def _get_loss_and_backward(self, batch):
        input, target = batch

        # Prune the weights per "args.prune_interval" if it's in the "prune mode"
        if self.do_prune:
            if self.last_epoch != self.cur_epoch and self.cur_epoch % self.args.prune_interval == 0:
                self.s_model_pruner.prune(self.args.prune_mode, self.args.prune_rates)
                self.last_epoch = self.cur_epoch
                print_nonzeros(self.s_model)

        # Get the total_loss and backward
        if self.do_dist:
            # Do different kinds of distillation according to "args.distill"
            s_feat, s_logit = self.s_model(input, is_group_feat=self.is_group, is_block_feat=self.is_block)
            t_feat, t_logit = self.t_model(input, is_group_feat=self.is_group, is_block_feat=self.is_block)
            s_f, t_f = self._get_dist_feat(self.args.distill, s_feat, t_feat)
            loss_cls = self.criterion_cls(s_logit, target)
            loss_div = self.criterion_div(s_logit, t_logit)
            loss_kd = self.criterion_kd(s_f, t_f)
            loss = loss_cls + loss_div * self.args.alpha + loss_kd * self.args.beta
        else:
            # Normal training
            s_logit = self.s_model(input)
            loss_cls = self.criterion_cls(s_logit, target)
            loss_div = loss_kd = torch.zeros(1).to(self.device)
            loss = loss_cls
        loss.backward()

        # Set the gradient of the pruned weights to 0 if it's in the "hard prune mode"
        if self.do_prune and not self.do_soft_prune:
            self._mask_prune_weight_grad()

        # Get performance metrics
        top1, top5 = accuracy(s_logit, target, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': loss.item(),
                'cls_loss': loss_cls.item(),
                'div_loss': loss_div.item(),
                'kd_loss': loss_kd.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, self.global_step
        )
        return loss, top1, top5

    def _evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.s_model(input_var)
        loss = self.criterion_cls(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}


def main():
    set_seeds(args.seed)
    check_dirs_exist([args.save_dir])
    logger = Logger(args.log_path)
    device = get_device()
    if args.dataset not in dataset.__dict__:
        raise NameError
    if args.model not in models.__dict__:
        raise NameError
    train_loader, eval_loader, num_classes = dataset.__dict__[args.dataset](args.batch_size)
    t_model = models.__dict__[args.model](num_classes=num_classes)
    s_model = models.__dict__[args.model](num_classes=num_classes)
    load_model(t_model, args.t_load_model_path, logger, device)
    load_model(s_model, args.s_load_model_path, logger, device)
    optimizer = optim.SGD(
        s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    base_trainer_cfg = (args, s_model, train_loader, eval_loader, optimizer, args.save_dir, device, logger)
    writer = SummaryWriter(log_dir=args.log_dir)  # For tensorboardX
    trainer = PMSPModelTrainer(t_model, writer, *base_trainer_cfg)
    logger.log('\n'.join(map(str, vars(args).items())))
    if args.evaluate:
        trainer.eval()
    else:
        trainer.train()
        if args.soft_prune:
            s_model.prune(args.prune_mode, args.prune_rates)
        trainer.eval()


if __name__ == '__main__':
    main()
