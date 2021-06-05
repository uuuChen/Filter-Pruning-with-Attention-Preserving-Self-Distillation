import argparse
import os
import sys
import math
import time
import numpy as np

from helpers.utils import (
    check_dirs_exist,
    get_device,
    accuracy,
    load_model,
    save_model,
    print_nonzeros,
    set_seeds,
    Logger
)
from helpers import dataset
from helpers.extractor import (
    FeatureExtractor,
    ConvWeightExtractor
)
from helpers.trainer import Trainer
from helpers.pruner import FiltersPruner
from helpers.distiller import FilterAttentionDistiller
import models

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser(description='Attention Distilled With Pruned Model Process')
parser.add_argument('--n-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--lr-drops', type=float, nargs='+', default=[0.1, 0.1, 0.1])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--leaky-relu-scope', type=float, default=0.2)
parser.add_argument('--prune-mode', type=str, default='None')  # No prune by default
parser.add_argument('--soft-prune', action='store_true', default=False)  # Do soft pruning or not
parser.add_argument('--prune-rates', nargs='+', type=float, default=[1.0])
parser.add_argument('--samp-batches', type=int, default=None)  # Sample batches to compute gradient for pruning. Use
# all batches by default
parser.add_argument('--use-actPR', action='store_true', default=False)  # Compute actual pruning rates for conv layers
# or not
parser.add_argument('--use-greedy', action='store_true', default=False)  # Prune filters by greedy or independent
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # Do pruning process once by default
parser.add_argument('--dist-mode', type=str, default='None')  # Pattern: "((all|conv|fc)(-attn)?(-grad)?-dist|None)"
parser.add_argument('--dist-method', type=str, default='attn-feature')
parser.add_argument('--dist-temperature', type=float, default=4.0)
parser.add_argument('--gad-factor', type=float, default=50.0)
parser.add_argument('--kd-factor', type=float, default=0.9)
parser.add_argument('--t-load-model-path', type=str, default='None')
parser.add_argument('--s-load-model-path', type=str, default='None')
parser.add_argument('--adapt-dim', type=int, default=128)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{int(time.time())}'
args.log_dir = f'{args.save_dir}/log'
args.log_path = f'saves/logs.txt'
if args.t_load_model_path is 'None':
    args.t_load_model_path = args.s_load_model_path


class PWADModelTrainer(Trainer):
    """  A trainer for weight distillation combined with attention mechanism and hard or soft pruning. """
    def __init__(self, t_model, adapt_dim, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_model = self.model
        self.t_model = t_model
        self.writer = writer

        self.do_prune = self.args.prune_mode is not 'None'
        self.do_soft_prune = self.args.soft_prune
        self.cross_entropy = nn.CrossEntropyLoss()

        self.s_pruner = FiltersPruner(
            self.s_model,
            self.optimizer,
            self.train_loader,
            self.logger,
            device=self.device,
            use_actPR=self.args.use_actPR,
            use_greedy=self.args.use_greedy,
            samp_batches=self.args.samp_batches,
        )
        self.conv_extractor = ConvWeightExtractor()
        self.s_weights, s_shapes = self._get_weight_for_dist(self.s_model)
        self.t_weights, t_shapes = self._get_weight_for_dist(self.t_model, n_dist_layers=len(s_shapes))
        self.w_distiller = FilterAttentionDistiller(
            s_shapes=s_shapes,
            t_shapes=t_shapes,
            adapt_dim=adapt_dim
        )
        self.optimizer.add_param_group({'params': self.w_distiller.parameters()})
        self.last_epoch = None

        self.t_model.eval()
        self.t_model = self.t_model.to(self.device)

    def _get_weight_for_dist(self, model, n_dist_layers=None):
        conv_dict = self.conv_extractor(model)
        n_layers = len(conv_dict)
        if n_dist_layers is None:
            n_dist_layers = n_layers

        def get_indices():
            nonlocal n_layers, n_dist_layers
            start = 0
            i = start
            ind = list()
            n_steps = math.ceil(n_layers / n_dist_layers)
            for _ in range(n_dist_layers):
                ind.append(i)
                i += n_steps
                if i >= n_layers:
                    start += 1
                    i = start
            ind = sorted(ind)
            return ind

        indices = get_indices()
        weights = list()
        shapes = list()
        for w in np.array(list(conv_dict.values()))[indices]:
            weights.append(w)
            shapes.append(w.shape)
        return weights, shapes

    def _mask_pruned_weight_grad(self):
        conv_mask = self.s_pruner.get_conv_mask()
        for name, module in self.s_model.named_modules():
            if name in conv_mask:
                grad = module.weight.grad
                ori_grad_arr = grad.data.cpu().numpy()
                new_grad_arr = ori_grad_arr * conv_mask[name]
                grad.data = torch.from_numpy(new_grad_arr).to(self.device)

    def _get_loss_and_backward(self, batch):
        input_var, target_var = batch

        # Prune the weights per "args.prune_interval" if it's in the "prune mode"
        if self.do_prune:
            if self.last_epoch != self.cur_epoch and self.cur_epoch % self.args.prune_interval == 0:
                self.s_pruner.prune(self.args.prune_mode, self.args.prune_rates)
                self.last_epoch = self.cur_epoch
                print_nonzeros(self.s_model)

        # 1. Distill the conv weight of the teacher to the student
        # 2. Forward and get the prediction loss
        # 3. Backward and pass the gradients to the distiller
        upd_s_w = self.w_distiller(self.s_weights, self.t_weights)
        s_output_var = self.s_model(input_var)
        pred_loss = self.cross_entropy(s_output_var, target_var)
        pred_loss.backward(retain_graph=True)
        for upd_w, ori_w in zip(upd_s_w, self.s_weights):
            upd_w.backward(ori_w.grad, retain_graph=True)

        # Set the gradient of the pruned weights to 0 if it's in the "hard prune mode"
        if self.do_prune and not self.do_soft_prune:
            self._mask_pruned_weight_grad()

        # Get performance metrics
        top1, top5 = accuracy(s_output_var, target_var, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, self.global_step
        )
        return pred_loss, top1, top5

    def _evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.s_model(input_var)
        loss = self.cross_entropy(output_var, target_var)
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
    logger.log_line()
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
    trainer = PWADModelTrainer(t_model, args.adapt_dim, writer, *base_trainer_cfg)
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
