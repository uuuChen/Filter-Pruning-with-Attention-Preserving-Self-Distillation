# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

# Modified to be integrated into the repository.

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFDBuilder():
    LAYER = {
        'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
        'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
        'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
        'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
        'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),  # 12
        'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
        'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
        'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
        'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
        'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
    }

    def __init__(self):
        pass

    def unique_shape(selg, s_shapes):
        n_s = []
        unique_shapes = []
        n = -1
        for s_shape in s_shapes:
            if s_shape not in unique_shapes:
                unique_shapes.append(s_shape)
                n += 1
            n_s.append(n)
        return n_s, unique_shapes

    def __call__(self, args, t_model, s_model):
        args.guide_layers = self.LAYER[args.t_model]
        args.hint_layers = self.LAYER[args.s_model]
        args.qk_dim = 128
        if args.dataset in ['cifar10', 'cifar100', 'cinic10']:
            image_size = 32
        else:
            image_size = 224
        data = torch.randn(2, 3, image_size, image_size)
        t_model.eval()
        s_model.eval()
        with torch.no_grad():
            feat_t, _ = t_model(data, is_block_feat=True)
            feat_s, _ = s_model(data, is_block_feat=True)
        args.s_shapes = [feat_s[i].size() for i in args.hint_layers]
        args.t_shapes = [feat_t[i].size() for i in args.guide_layers]
        args.n_t, args.unique_t_shapes = self.unique_shape(args.t_shapes)
        return AFD(args)
    
    
class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        loss = self.attention(g_s, g_t)
        return sum(loss)
    

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s
