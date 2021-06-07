import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSimilarity(nn.Module):
    """Multiple layers of similarity-preserving knowledge Distillation"""
    def __init__(self):
        super().__init__()

    def forward(self, s_g, t_g):
        # --------------------------------------------
        # Shape of s_g : (s_nl,), (bs, s_ch, s_h, s_w)
        # Shape of t_g : (t_nl,), (bs, t_ch, t_h, t_w)
        # --------------------------------------------
        s_nl = len(s_g)
        t_nl = len(t_g)

        s_mtx = torch.stack([self.get_sim_matrix_v2(s_f) for s_f in s_g])  # (s_nl, bs, bs)
        t_mtx = torch.stack([self.get_sim_matrix_v2(t_f) for t_f in t_g])  # (t_nl, bs, bs)

        s_mtx = torch.unsqueeze(s_mtx, dim=1).repeat(1, t_nl, 1, 1)  # (s_nl, t_nl, bs, bs)
        t_mtx = torch.unsqueeze(t_mtx, dim=0).repeat(s_nl, 1, 1, 1)  # (s_nl, t_nl, bs, bs)

        loss = (s_mtx - t_mtx).pow(2).view(s_nl, -1).mean(1).mean()  # (1,)
        return loss

    def get_sim_matrix(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        f = f.view(f.shape[0], -1)  # (bs, ch * h * w)
        n_f = F.normalize(f, dim=1)  # (bs, ch * h * w)
        mtx = torch.matmul(n_f, torch.t(n_f))  # (bs, bs)
        n_mtx = F.normalize(mtx, dim=1)  # (bs, bs)
        return n_mtx

    def get_sim_matrix_v2(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        a_f = self.at(f)  # (bs, h * w)
        mtx = torch.matmul(a_f, torch.t(a_f))  # (bs, bs)
        n_mtx = F.normalize(mtx, dim=1)  # (bs, bs)
        return n_mtx

    def at(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)
        # --------------------------------------------
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))  # (bs, h * w)
