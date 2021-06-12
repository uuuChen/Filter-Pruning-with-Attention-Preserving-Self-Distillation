import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g : (s_nl,), (bs, s_ch, s_h, s_w)
        # Shape of t_g : (t_nl,), (bs, t_ch, t_h, t_w)
        # --------------------------------------------
        loss = sum([self.similarity_loss(s_f, t_f) for s_f, t_f in zip(s_g, t_g)])  # (1,)
        return loss

    def similarity_loss(self, s_f, t_f):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_w)
        # Shape of t_f : (bs, t_ch, t_h, t_w)
        # --------------------------------------------
        bs = s_f.shape[0]
        s_f = s_f.view(bs, -1)  # (bs, s_ch * s_h * s_w)
        t_f = t_f.view(bs, -1)  # (bs, t_ch * t_h * t_w)

        s_g = torch.mm(s_f, torch.t(s_f))  # (bs, bs)
        s_g = torch.nn.functional.normalize(s_g)  # (bs, bs)
        t_g = torch.mm(t_f, torch.t(t_f))  # (bs, bs)
        t_g = torch.nn.functional.normalize(t_g)  # (bs, bs)

        g_diff = t_g - s_g  # (bs, bs)
        loss = (g_diff * g_diff).view(-1, 1).sum(0) / (bs * bs)  # (1,)
        return loss
