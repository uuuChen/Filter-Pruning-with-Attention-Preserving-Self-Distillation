import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAttention(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g (group) : (s_nl,), (bs, s_ch, s_h, s_h)
        # Shape of t_g (group) : (t_nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        loss = torch.mean(torch.stack([self.s_to_all_t_loss(s_f, t_g) for s_f in s_g]))  # (1,)
        return loss

    def s_to_all_t_loss(self, s_f, t_g):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_h)
        # Shape of t_g : (t_nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        d = dict()
        vals = torch.stack([self.at_loss(self.s_sample(s_f, t_f, d), t_f) for t_f in t_g])  # (t_nl,)
        d_vals = vals.detach()
        atts = F.softmax(-1 * d_vals / torch.mean(d_vals), dim=0)  # (t_nl,)
        loss = torch.mean(vals * atts)  # (1,)
        return loss

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
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))  # (bs, w * w)
