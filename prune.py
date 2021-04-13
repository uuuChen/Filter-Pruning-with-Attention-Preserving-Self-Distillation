import argparse
import os
import sys
import numpy as np

from util import check_dirs_exist, get_device, accuracy, load_model, save_model, print_nonzeros, set_seeds
from data_loader import DataLoader
from models.alexnet import alexnet
from models.feature_extractor import FeatureExtractor
from trainer import Trainer

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser(description='Prune Process')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_drop', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100, 150])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--prune-mode', type=str, default=None)
parser.add_argument('--prune-rates', nargs='+', type=float, default=[0.16, 0.62, 0.65, 0.63, 0.63])
parser.add_argument('--prune-interval', type=int, default=sys.maxsize)  # By default we will only prune once
parser.add_argument('--dist-mode', type=str, default=None)  # pattern: "((all|conv|fc)(-grad)?-dist|None)"
parser.add_argument('--t-load-model-path', type=str, default=None)
parser.add_argument('--s-load-model-path', type=str, default=None)
parser.add_argument('--adapt-hidden-size', type=int, default=128)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # For Mac OS
args.save_dir = f'saves/{args.model}_{args.dataset}/{args.prune_mode}'
args.save_dir += '-once' if args.prune_interval == sys.maxsize else f'-{args.prune_interval}'
args.save_dir += f'/{args.lr}'
args.log_dir = os.path.join(args.save_dir, 'log')
if args.t_load_model_path is None:
    args.t_load_model_path = args.s_load_model_path


class PGADModelTrainer(Trainer):
    """  A trainer for gradually self-distillation combined with attention mechanism and hard or soft pruning. """
    def __init__(self,
                 t_model,
                 adapt_hidden_size,
                 writer,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.t_model = t_model
        self.s_model = self.model
        self.adapt_hidden_size = adapt_hidden_size
        self.writer = writer

        self.do_prune = self.args.prune_mode is not None
        self.do_dist = self.args.dist_mode is not None
        self.cross_entropy = nn.CrossEntropyLoss()
        self.filter_prune_rates = self.s_model.get_filters_prune_rates(self.args.prune_rates)
        self.t_model_with_FE = FeatureExtractor(self.t_model)
        self.s_model_with_FE = FeatureExtractor(self.s_model)

        self.last_epoch = None
        self.init_adapt_layers = False

    def zeroize_pruned_weights_grad(self):
        for p in self.s_model.parameters():
            tensor_arr = p.data.cpu().numpy()
            ori_grad_arr = p.grad.data.cpu().numpy()
            new_grad_arr = np.where(tensor_arr == 0, 0, ori_grad_arr)
            p.grad.data = torch.from_numpy(new_grad_arr).to(self.device)

    def trans_features_for_dist(self, features_dict):
        def get_conv_attn_feature(feature):
            return F.normalize(torch.sum(torch.pow(feature, 2), dim=1)).view(feature.shape[0], -1)

        dist_features = list()
        for name, feature in features_dict.items():
            if 'conv' in name and 'fc' not in self.args.dist_mode:
                dist_features.append(get_conv_attn_feature(feature))
            elif 'fc' in name and 'conv' not in self.args.dist_mode:
                dist_features.append(feature)
        return dist_features

    def build_adapt_layers_for_atten(self, s_model, dist_features):
        # We only need to use the shape of the features, so we don't need to pass student's and teacher's features to
        # this function at the same time
        adapt_layers = list()
        for feature in dist_features:
            adapt_layers.append(nn.Linear(feature.shape[1], self.args.adapt_hidden_size))
        s_model.adapt_layers = nn.ModuleList(adapt_layers)
        s_model.adapt_layers.to(self.device)
        self.optimizer = optim.SGD(
            self.s_model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

    def get_GAD_loss(self, s_dist_features, t_dist_features):
        dist_losses = list()
        for s_feature, t_feature in zip(s_dist_features, t_dist_features):
            print(s_feature.shape, t_feature.shape)
        print(len(s_dist_features), len(t_dist_features))
        raise Exception

    def get_loss_and_backward(self, batch, global_step):
        input_var, target_var = batch

        # Prune the weights per "args.prune_interval"
        if self.do_prune:
            if self.last_epoch != self.cur_epoch and self.cur_epoch % self.args.prune_interval == 0:
                self.last_epoch = self.cur_epoch

                # In order to get the gradient and use it on the criterion "filter-ga"
                s_output_var = self.s_model(input_var)
                s_loss = self.cross_entropy(s_output_var, target_var)
                s_loss.backward(retain_graph=True)

                # Prune the weights by "args.prune_mode"
                self.s_model.prune(self.args.prune_mode, self.args.prune_rates)

        # ---------------------------------------------
        # 1. Get the output of each layers of student's and teacher's model and transform them for distillation
        # 2. If "self.init_adapt_layers" is "False", initialize the adaption layers for the attention mechanism
        # 3. Do the gradually self-distillation combined with attention mechanism
        # 4. Get loss and do backward, and then set the gradient of the pruned weights to 0 if it's in the "hard"
        #    prune mode
        # ---------------------------------------------
        s_output_var, s_features_dict = self.s_model_with_FE(input_var)
        t_output_var, t_features_dict = self.t_model_with_FE(input_var)
        if self.do_dist:
            s_dist_features = self.trans_features_for_dist(s_features_dict)
            t_dist_features = self.trans_features_for_dist(t_features_dict)
            if not self.init_adapt_layers:
                self.init_adapt_layers = True
                self.build_adapt_layers_for_atten(self.s_model, s_dist_features)
            pred_loss = self.cross_entropy(s_output_var, target_var)
            GAD_loss = self.get_GAD_loss(s_dist_features, t_dist_features)
        else:
            pred_loss = self.cross_entropy(s_output_var, target_var)
            GAD_loss = torch.zeros(1)
        total_loss = pred_loss + GAD_loss
        total_loss.backward()

        if 'hard' in self.args.prune_mode:
            self.zeroize_pruned_weights_grad()

        # Get performance metrics
        top1, top5 = accuracy(s_output_var, target_var, topk=(1, 5))
        self.writer.add_scalars(
            'data/scalar_group', {
                'total_loss': total_loss.item(),
                'pred_loss': pred_loss.item(),
                'gad_loss': GAD_loss.item(),
                'lr': self.cur_lr,
                'top1': top1,
                'top5': top5
            }, global_step
        )
        return total_loss, top1, top5

    def evaluate(self, batch):
        input_var, target_var = batch
        output_var = self.s_model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        top1, top5 = accuracy(output_var, target_var, topk=(1, 5))
        return {'loss': loss, 'top1': top1, 'top5': top5}


def main():
    set_seeds(args.seed)
    check_dirs_exist([args.save_dir])
    device = get_device()
    if args.dataset == 'cifar100':
        train_loader, eval_loader = DataLoader.get_cifar100(args.batch_size)  # get data loader
        num_classes = 100
    else:
        raise ValueError
    if args.model == 'alexnet':
        t_model = alexnet(num_classes=num_classes)
        s_model = alexnet(num_classes=num_classes)
        load_model(t_model, args.t_load_model_path, device)
        load_model(s_model, args.s_load_model_path, device)
    else:
        raise ValueError
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    base_trainer_cfg = (args, s_model, train_loader, eval_loader, optimizer, args.save_dir, device)
    writer = SummaryWriter(log_dir=args.log_dir)  # for tensorboardX
    trainer = PGADModelTrainer(
        t_model,
        args.adapt_hidden_size,
        writer,
        *base_trainer_cfg
    )
    # trainer.eval()  # Show loaded model performance as baseline
    trainer.train()
    if 'soft' in args.prune_mode:
        s_model.prune(args.prune_mode, args.prune_rates)
    trainer.eval()


if __name__ == '__main__':
    main()
