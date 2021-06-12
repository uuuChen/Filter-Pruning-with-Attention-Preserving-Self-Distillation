import argparse
import os
import sys
import math
import time

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
import models
from helpers.extractor import FeatureExtractor
from helpers.trainer import Trainer
from helpers.pruner import FiltersPruner

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
parser.add_argument('--prune-mode', type=str, default='None')
parser.add_argument('--prune-rates', nargs='+', type=float, default=[1.0])  # No prune by default
parser.add_argument('--samp-batches', type=int, default=None)  # Sample batches to compute gradient for pruning. Use
# all batches by default
parser.add_argument('--use-actPR', action='store_true', default=False)  # Compute actual pruning rates for conv layers
# or not
parser.add_argument('--use-greedy', action='store_true', default=False)  # Prune filters by greedy or independent
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # We will only prune once by default
parser.add_argument('--dist-mode', type=str, default='None')  # pattern: "((all|conv|fc)(-attn)?(-grad)?-dist|None)"
parser.add_argument('--dist-method', type=str, default='attn-feature')
parser.add_argument('--dist-temperature', type=float, default=4.0)
parser.add_argument('--gad-factor', type=float, default=50.0)
parser.add_argument('--kd-factor', type=float, default=0.9)
parser.add_argument('--t-load-model-path', type=str, default='None')
parser.add_argument('--s-load-model-path', type=str, default='None')
parser.add_argument('--adapt-hidden-size', type=int, default=128)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{args.model}_{args.dataset}/{args.prune_mode}'
args.save_dir += '-once' if args.prune_interval == sys.maxsize else f'-{args.prune_interval}'
args.save_dir += f'/{args.dist_mode}/{int(time.time())}'
args.log_dir = os.path.join(args.save_dir, 'log')
args.log_path = os.path.join(args.save_dir, "logs.txt")
if args.t_load_model_path is 'None':
    args.t_load_model_path = args.s_load_model_path


class PGADModelTrainer(Trainer):
    """  A trainer for gradually self-distillation combined with attention mechanism and hard or soft pruning. """
    def __init__(self, t_model, adapt_hidden_size, writer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_model = t_model
        self.s_model = self.model
        self.adapt_hidden_size = adapt_hidden_size
        self.writer = writer

        self.do_prune = self.args.prune_mode is not 'None'
        self.do_dist = self.args.dist_mode is not 'None'
        self.do_soft_prune = 'soft' in self.args.prune_mode
        self.do_attn_dist = 'attn' in self.args.dist_mode
        self.do_grad_dist = 'grad' in self.args.dist_mode
        self.do_conv_dist = 'conv' in self.args.dist_mode
        self.do_fc_dist = 'fc' in self.args.dist_mode
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')  # Not sure for using "batchmean"
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.args.leaky_relu_scope)

        self.t_model_with_FE = FeatureExtractor(self.t_model)
        self.s_model_with_FE = FeatureExtractor(self.s_model)
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
        self.init_adapt_layers = False

        self.t_model.eval()
        self.t_model = self.t_model.to(self.device)

    def _get_GA_coefs(self, s_dist_features, t_dist_features):
        def get_n_grad_dist(cur_epoch, n_epochs, n_all_dist):
            return min(math.ceil(((cur_epoch + 1) / n_epochs) * n_all_dist * 2), n_all_dist)

        def get_n_grad_dist_v2(cur_epoch, schedule, n_epochs, n_all_dist):
            sc = [0] + schedule + [n_epochs]
            i = None
            for i in range(len(sc)):
                if cur_epoch < sc[i]:
                    break
            range_epochs = sc[i] - sc[i-1]
            return min(math.ceil(((cur_epoch + 1 - sc[i-1]) / range_epochs) * n_all_dist * 2), n_all_dist)

        def get_attn_scores(pair_features):
            scores = list()
            for i, (s_feature, t_feature) in enumerate(pair_features):
                score = torch.mean(torch.abs(t_feature.detach() - s_feature.detach()))
                scores.append(score)
            scores = torch.stack(scores, dim=0)
            scores = scores / torch.sum(scores)
            return scores

        n_all_dist = len(s_dist_features)
        if self.do_grad_dist and self.do_attn_dist:
            n_grad_dist = get_n_grad_dist(self.cur_epoch, self.args.n_epochs, n_all_dist)
            pair_feats = list(zip(s_dist_features, t_dist_features))[:n_grad_dist]
            attn_scores = get_attn_scores(pair_feats)
            dist_coefs = torch.zeros(n_all_dist, dtype=torch.float64).to(self.device)
            dist_coefs[:n_grad_dist] = attn_scores
        elif self.do_attn_dist:
            pair_feats = list(zip(s_dist_features, t_dist_features))
            dist_coefs = get_attn_scores(pair_feats)
        elif self.do_grad_dist:
            n_grad_dist = get_n_grad_dist(self.cur_epoch, self.args.n_epochs, n_all_dist)
            dist_coefs = torch.zeros(n_all_dist, dtype=torch.float64).to(self.device)
            dist_coefs[:n_grad_dist] = 1 / n_grad_dist
            # n_grad_dist = get_n_grad_dist_v2(self.cur_epoch, self.args.schedule, self.args.n_epochs, n_all_dist)
            # dist_coefs = torch.zeros(n_all_dist, dtype=torch.float64).to(self.device)
            # dist_coefs[:n_grad_dist] = 1 / n_all_dist
        else:
            dist_coefs = torch.ones(n_all_dist, dtype=torch.float64).to(self.device)
            dist_coefs /= n_all_dist
        return dist_coefs

    def _mask_pruned_weight_grad(self):
        conv_mask = self.s_model_pruner.conv_mask
        for name, module in self.s_model.named_modules():
            if name in conv_mask:
                grad = module.weight.grad
                ori_grad_arr = grad.data.cpu().numpy()
                new_grad_arr = ori_grad_arr * conv_mask[name]
                grad.data = torch.from_numpy(new_grad_arr).to(self.device)

    def _trans_features_for_dist(self, features_dict):
        def get_conv_attn_feature(feature):
            return F.normalize(torch.sum(torch.pow(feature, 2), dim=1).view(feature.shape[0], -1), dim=1)

        def get_flat_norm_feature(feature):
            return F.normalize(feature.view(feature.shape[0], -1), dim=1)

        dist_features = list()
        for i, feature in enumerate(features_dict.values(), start=1):
            if i != len(features_dict):
                if len(feature.shape) == 4 and not self.do_fc_dist:  # Conv layer
                    if self.args.dist_method == 'attn-feature':
                        dist_feature = get_conv_attn_feature(feature)
                    elif self.args.dist_method == 'flat-feature':
                        dist_feature = get_flat_norm_feature(feature)
                    else:
                        raise NameError
                elif len(feature.shape) == 2 and not self.do_conv_dist:  # Fc layer
                    dist_feature = feature
                else:
                    continue
            else:  # Logit
                dist_feature = feature
            dist_features.append(dist_feature)
        return dist_features

    def _get_GAD_loss(self, s_dist_features, t_dist_features):
        # Get feature loss of all distilled layers
        feature_losses = list()
        for s_feature, t_feature in zip(s_dist_features, t_dist_features):
            feature_losses.append(self.mse_loss(s_feature, t_feature.detach()))
        feature_losses = torch.stack(feature_losses)

        # Combine feature losses and soft logit loss with attention coefficients
        GA_coefs = self._get_GA_coefs(s_dist_features, t_dist_features)
        GAD_loss = torch.mean(feature_losses * GA_coefs)

        # Print (loss, coefficient) pairs
        # loss_coef_pairs = list(zip(feature_losses.cpu().detach().numpy(), GA_coefs.cpu().detach().numpy()))
        # print(loss_coef_pairs)

        return GAD_loss

    def _get_KD_loss(self, s_logit, t_logit):
        T = self.args.dist_temperature
        loss = self.kl_div(
            F.log_softmax(s_logit / T, dim=1),
            F.softmax(t_logit.detach() / T, dim=1),
        ) * T * T
        return loss

    def _get_loss_and_backward(self, batch):
        input, target = batch

        # Prune the weights per "args.prune_interval" if it's in the "prune mode"
        if self.do_prune:
            if self.last_epoch != self.cur_epoch and self.cur_epoch % self.args.prune_interval == 0:
                self.last_epoch = self.cur_epoch
                self.s_model_pruner.prune(self.args.prune_mode, self.args.prune_rates)
                print_nonzeros(self.s_model)

        # Do different kinds of distillation according to "dist_mode" if "do_dist", otherwise do general
        # training
        s_logit, s_features_dict = self.s_model_with_FE(input)
        t_logit, t_features_dict = self.t_model_with_FE(input)
        if self.do_dist:
            s_dist_features = self._trans_features_for_dist(s_features_dict)
            t_dist_features = self._trans_features_for_dist(t_features_dict)
            pred_loss = self.cross_entropy(s_logit, target)
            GAD_loss = self._get_GAD_loss(s_dist_features[:-1], t_dist_features[:-1])
            KD_loss = self._get_KD_loss(s_dist_features[-1], t_dist_features[-1])
        else:
            pred_loss = self.cross_entropy(s_logit, target)
            GAD_loss = KD_loss = torch.zeros(1, dtype=torch.float64).to(self.device)
        total_loss = pred_loss + GAD_loss * self.args.gad_factor + KD_loss * self.args.kd_factor
        total_loss.backward()

        # Set the gradient of the pruned weights to 0 if it's in the "hard prune mode"
        if self.do_prune and not self.do_soft_prune:
            self._mask_pruned_weight_grad()

        # Get performance metrics
        top1, top5 = accuracy(s_logit, target, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': total_loss.item(),
                'pred_loss': pred_loss.item(),
                'gad_loss': GAD_loss.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, self.global_step
        )
        return total_loss, top1, top5

    def _evaluate(self, batch):
        input, target = batch
        logit = self.s_model(input)
        loss = self.cross_entropy(logit, target)
        top1, top5 = accuracy(logit, target, topk=(1, 5))
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
    trainer = PGADModelTrainer(t_model, args.adapt_hidden_size, writer, *base_trainer_cfg)
    logger.log('\n'.join(map(str, vars(args).items())))
    if args.evaluate:
        trainer.eval()
    else:
        trainer.train()
        if 'soft' in args.prune_mode:
            s_model.prune(args.prune_mode, args.prune_rates)
        trainer.eval()
    print(f'Log Path : {args.log_path}')


if __name__ == '__main__':
    main()
