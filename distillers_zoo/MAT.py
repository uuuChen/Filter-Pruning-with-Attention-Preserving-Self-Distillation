import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiAttention(nn.Module):
    def __init__(self, window_size=None):
        super().__init__()
        self.w_s = window_size  # Size of the window
        self.l_r = None

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g (group) : (s_nl,), (bs, s_ch, s_h, s_h)
        # Shape of t_g (group) : (t_nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        self.l_r = math.ceil(len(t_g) / len(s_g))
        loss = torch.mean(torch.stack([self.s_to_all_t_loss(s_f, t_g, i) for i, s_f in enumerate(s_g)]))  # (1,)
        return loss

    def s_to_all_t_loss(self, s_f, t_g, s_idx):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_h)
        # Shape of t_g : (nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        d = dict()
        vals = torch.stack([self.at_loss(self.s_sample(s_f, t_f, d), t_f) for t_f in self.t_sample(t_g, s_idx)])  # (
        # w_s,)
        d_vals = vals.detach()
        atts = F.softmax(-1 * d_vals / torch.mean(d_vals), dim=0)  # (w_s,)
        loss = torch.mean(vals * atts)  # (1,)
        return loss

    def t_sample(self, t_g, s_idx):
        # --------------------------------------------
        # Shape of r : (1,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        if not self.w_s:
            return t_g
        l = self.l_r * s_idx
        r = l + self.w_s
        l = max(l, 0)
        r = min(r, len(t_g))
        return t_g[l:r]

    def s_sample(self, s_f, t_f, d):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_h)
        # Shape of t_f : (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        t_h = t_f.shape[2]
        if t_h not in d:  # Reuse the result of "adaptive_avg_pool2d" For speed optimization
            d[t_h] = F.adaptive_avg_pool2d(s_f, (t_h, t_h))  # (bs, s_ch, t_h, t_h)
        return d[t_h]

    def at_loss(self, s_f, t_f):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, t_h, t_h)
        # Shape of t_f : (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        return (self.at(s_f) - self.at(t_f)).pow(2).mean()  # (1,)

    def at(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, w, w)
        # --------------------------------------------
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))  # (bs, w * w)
