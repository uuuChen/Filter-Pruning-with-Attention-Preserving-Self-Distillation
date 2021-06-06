from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g (group) : (nl,), (bs, s_ch, s_w, s_w)
        # Shape of t_g (group) : (nl,), (bs, t_ch, t_w, t_w)
        # --------------------------------------------
        loss = torch.sum(torch.stack([self.at_loss(s_f, t_f) for s_f, t_f in zip(s_g, t_g)]))  # (1,)
        return loss

    def at_loss(self, s_f, t_f):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_w, s_w)
        # Shape of t_f : (bs, t_ch, t_w, t_w)
        # --------------------------------------------
        s_w, t_w = s_f.shape[2], t_f.shape[2]
        if s_w > t_w:
            s_f = F.adaptive_avg_pool2d(s_f, (t_w, t_w))
        elif s_w < t_w:
            t_f = F.adaptive_avg_pool2d(t_f, (s_w, s_w))
        else:
            pass
        return (self.at(s_f) - self.at(t_f)).pow(2).mean()

    def at(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, w, w)
        # --------------------------------------------
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))  # (bs, w * w)
