from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer
    """
    def __init__(self, p=2, dataset=None):
        super(Attention, self).__init__()
        self.p = p
        self.dataset = dataset

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g (group) : (nl,), (bs, s_ch, s_h, s_h)
        # Shape of t_g (group) : (nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        loss = sum([self.at_loss(s_f, t_f) for s_f, t_f in zip(s_g, t_g)])  # (1,)
        return loss

    def at_loss(self, s_f, t_f):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_h)
        # Shape of t_f : (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        s_h, t_h = s_f.shape[2], t_f.shape[2]
        if s_h > t_h:
            s_f = F.adaptive_avg_pool2d(s_f, (t_h, t_h))
        elif s_h < t_h:
            t_f = F.adaptive_avg_pool2d(t_f, (s_h, s_h))
        else:
            pass
        return (self.at(s_f) - self.at(t_f)).pow(2).mean()

    def at(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, h)
        # --------------------------------------------
        if self.dataset == 'imagenet':
            return F.normalize(f.pow(self.p).sum(1).view(f.size(0), -1))  # (bs, h * h)
        else:
            return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))  # (bs, h * h)
