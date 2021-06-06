import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, s_g, t_g):
        return torch.sum(torch.stack([self.similarity_loss(s_f, t_f) for s_f, t_f in zip(s_g, t_g)]))

    def similarity_loss(self, s_f, t_f):
        bs = s_f.shape[0]
        s_f = s_f.view(bs, -1)
        t_f = t_f.view(bs, -1)

        s_g = torch.mm(s_f, torch.t(s_f))
        s_g = torch.nn.functional.normalize(s_g)
        t_g = torch.mm(t_f, torch.t(t_f))
        t_g = torch.nn.functional.normalize(t_g)

        g_diff = t_g - s_g
        loss = (g_diff * g_diff).view(-1, 1).sum(0) / (bs * bs)
        return loss
