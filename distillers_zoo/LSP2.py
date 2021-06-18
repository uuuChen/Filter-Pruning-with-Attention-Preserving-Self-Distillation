import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitSimilarity2(nn.Module):
    def __init__(self, window_size=None):
        super().__init__()
        self.w_s = window_size  # Size of the window

    def forward(self, s_g_l, t_g_l):
        # --------------------------------------------
        # Shape of s_g_l (group) : (nl,), (bs, s_ch, s_h, s_h) , (1,)
        # Shape of t_g_l (group) : (nl,), (bs, t_ch, t_h, t_h) , (1,)
        # --------------------------------------------
        s_g, s_l = s_g_l  # Group of student's features, student's logit
        t_g, t_l = t_g_l  # Group of teacher's features, teacher's logit
        loss = torch.stack([self.s_to_t_loss(s_f, t_g, i) for i, s_f in enumerate(s_g)]).mean()  # (1,)
        return loss

    def s_to_t_loss(self, s_f, t_g, s_idx):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, s_h, s_h)
        # Shape of t_g : (nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        loss = torch.stack([self.sim_loss(s_f, t_f) for t_f in self.t_sample(t_g, s_idx)]).mean()  # (1)
        return loss

    def t_sample(self, t_g, s_idx):
        # --------------------------------------------
        # Shape of t_g : (nl,), (bs, t_ch, t_h, t_h)
        # --------------------------------------------
        if not self.w_s:
            return t_g
        l = np.clip(s_idx - (self.w_s - 1) // 2, 0, None)
        r = np.clip(l + self.w_s, None, len(t_g))
        return t_g[l:r]

    def sim_loss(self, s_f, t_f, is_at=True):
        # --------------------------------------------
        # Shape of s_f : (bs, s_ch, t_h, t_h)  or (bs, n_class)
        # Shape of t_f : (bs, t_ch, t_h, t_h)  or (bs, n_class)
        # --------------------------------------------
        s_mtx = self.sim(s_f, is_at=is_at)
        t_mtx = self.sim(t_f, is_at=is_at)
        loss = (s_mtx - t_mtx).pow(2).mean()  # (1,)
        return loss

    def sim(self, f, is_at=True):
        # --------------------------------------------
        # Shape of f : (bs, ch, h, w)  or  (bs, n_class)
        # --------------------------------------------
        if is_at:
            f = self.at(f)
        f = f.view(f.shape[0], -1)  # (bs, ch * h * w) or (bs, n_class)
        f = F.normalize(f, dim=1)  # (bs, ch * h * w) or (bs, n_class)
        mtx = torch.matmul(f, torch.t(f))  # (bs, bs)
        mtx = F.normalize(mtx, dim=1)  # (bs, bs)
        return mtx

    def at(self, f):
        # --------------------------------------------
        # Shape of f : (bs, ch, w, w)
        # --------------------------------------------
        if len(f.shape) != 4:
            return f
        a_f = F.normalize(f.pow(2).mean(1).view(f.size(0), -1))  # (bs, w * w)
        return a_f
