import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


__all__ = ['FilterAttentionDistiller', 'KLDistiller']


class Add(Function):
    @staticmethod
    def forward(ctx, s_w, attn_w):
        ctx.a_shape = attn_w.shape
        s_w.data.add_(attn_w.view(s_w.shape))
        return s_w

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output.view(ctx.a_shape)


class FilterAttentionDistiller(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention = FilterAttention(**kwargs)

    def _update_w(self, s_w, attn_w):
        # --------------------------------------------
        # Shape of s_w    : (nl,), (s_nk, s_ch, s_w, s_h)
        # Shape of attn_w : (nl,), (s_nk, s_ch * s_w * s_h)
        # --------------------------------------------
        upd_w = list()
        for s, a in zip(s_w, attn_w):
            upd_w.append(Add.apply(s, a))
        return upd_w

    def forward(self, s_w, t_w):
        attn_w = self.attention(s_w, t_w)
        upd_w = self._update_w(s_w, attn_w)
        return upd_w


class FilterAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.s_shapes = kwargs.pop('s_shapes', None)  # (nl,), (4)
        self.t_shapes = kwargs.pop('t_shapes', None)  # (nl,), (4)
        adapt_dim = kwargs.pop('adapt_dim', 128)
        assert self.s_shapes is not None, f"Need to pass in args 's_shapes'"
        assert self.t_shapes is not None, f"Need to pass in args 't_shapes'"

        n_layers = len(self.s_shapes)
        self.s_enc_trans = EncodingTransform(self.s_shapes, adapt_dim)
        self.t_enc_trans = EncodingTransform(self.t_shapes, adapt_dim)
        self.t_dec_trans = DecodingTransform(adapt_dim, self.s_shapes)

        # Shape of adapt_params : (nl,), (2 * adapt_dim, 1)
        self.adapt_params = nn.ParameterList([nn.Parameter(torch.Tensor(2 * adapt_dim, 1)) for _ in range(n_layers)])
        for i in range(n_layers):
            torch.nn.init.xavier_normal_(self.adapt_params[i])

    def forward(self, s_w, t_w):
        # --------------------------------------------
        # Shape of s_w : (nl,), (s_nk, s_ch, s_w, s_h)
        # Shape of t_w : (nl,), (t_nk, t_ch, t_w, t_h)
        # --------------------------------------------

        # Shape of s_key : (nl,), (s_nk, adapt_dim)
        # Shape of t_qry : (nl,), (t_nk, adapt_dim)
        # Shape of value : (nl,), (t_nk, s_ch * s_w * s_h)
        s_key = self.s_enc_trans(s_w)
        t_qry = self.t_enc_trans(t_w)
        value = self.t_dec_trans(t_qry)

        # Shape of s_key : (nl,), (s_nk, t_nk, adapt_dim)
        # Shape of t_qry : (nl,), (s_nk, t_nk, adapt_dim)
        # Shape of pair  : (nl,), (s_nk, t_nk, 2 * adapt_dim)
        s_key = [torch.unsqueeze(key, dim=1).repeat(1, shape[0], 1) for key, shape in zip(s_key, self.t_shapes)]
        t_qry = [torch.unsqueeze(qry, dim=0).repeat(shape[0], 1, 1) for qry, shape in zip(t_qry, self.s_shapes)]
        pair = [torch.cat((key, qry), dim=2) for key, qry in zip(s_key, t_qry)]

        # Shape of logit  : (nl,), (s_nk, t_nk)
        # Shape of attn   : (nl,), (s_nk, t_nk)
        # Shape of attn_w : (nl,), (s_nk, s_ch * s_w * s_h)
        logit = [torch.einsum("sta,ao->st", _pair, _a) for _pair, _a in zip(pair, self.adapt_params)]
        attn = [F.softmax(F.leaky_relu(_logit, negative_slope=0.2), dim=1) for _logit in logit]
        attn_w = [torch.matmul(_attn, _val) for _attn, _val in zip(attn, value)]
        return attn_w


class LinearWithAct(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn=False, relu=False):
        out = self.linear(x)
        if bn:
            out = self.bn(out)
        if relu:
            out = self.relu(out)
        return out


class EncodingTransform(nn.Module):
    def __init__(self, shapes, adapt_dim):
        super().__init__()
        self.transforms = nn.ModuleList([LinearWithAct(np.prod(shape[1:]), adapt_dim) for shape in shapes])

    def forward(self, w):
        # --------------------------------------------
        # Shape of w : (nl,), (nk, ch, w, h)
        # --------------------------------------------
        w = [trans(_w.view(_w.shape[0], -1)) for trans, _w in zip(self.transforms, w)]  # (nl,), (nk, adapt_dim)
        return w


class DecodingTransform(nn.Module):
    def __init__(self, adapt_dim, shapes):
        super().__init__()
        self.transforms = nn.ModuleList([LinearWithAct(adapt_dim, np.prod(shape[1:])) for shape in shapes])

    def forward(self, w):
        # --------------------------------------------
        # Shape of w : (nl,), (nk, adapt_dim)
        # --------------------------------------------
        w = [trans(_w) for trans, _w in zip(self.transforms, w)]  # (nl,), (nk, ch * w * h)
        return w


class KLDistiller(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, s_logit, t_logit):
        loss = self.kl_div(
            F.log_softmax(s_logit / self.T, dim=1),
            F.softmax(t_logit.detach() / self.T, dim=1),
        ) * self.T * self.T
        return loss



