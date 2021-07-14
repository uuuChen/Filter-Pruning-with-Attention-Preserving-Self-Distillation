import torch
import torch.nn as nn
import torch.nn.functional as F


class AttenSimilarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(AttenSimilarity, self).__init__()

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g : (nl,), (bs, s_ch, s_h, s_w)
        # Shape of t_g : (nl,), (bs, t_ch, t_h, t_w)
        # --------------------------------------------
        loss = sum([self.asp_loss(s_f, t_f) for s_f, t_f in zip(s_g, t_g)])  # (1,)
        return loss

    def asp_loss(self, s_f, t_f):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_w)
        # Shape of t_f : (bs, t_ch, t_h, t_w)
        # --------------------------------------------
        bs = s_f.shape[0]
        s_f = self.at(s_f)  # (bs, s_h * s_w)
        t_f = self.at(t_f)  # (bs, t_h * t_w)

        s_g = torch.mm(s_f, torch.t(s_f))  # (bs, bs)
        s_g = torch.nn.functional.normalize(s_g)  # (bs, bs)
        t_g = torch.mm(t_f, torch.t(t_f))  # (bs, bs)
        t_g = torch.nn.functional.normalize(t_g)  # (bs, bs)

        g_diff = t_g - s_g  # (bs, bs)
        loss = (g_diff * g_diff).view(-1, 1).sum(0) / (bs * bs)  # (1,)
        return loss

    def at(self, f, is_flat=True):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        out = F.normalize(f.pow(2).mean(1))  # (bs, h, w)
        if is_flat:
            out = out.view(f.size(0), -1)  # (bs, h * w)
        return out
