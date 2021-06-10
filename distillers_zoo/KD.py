import torch.nn as nn
import torch.nn.functional as F


class KLDistiller(nn.Module):
    def __init__(self, T):
        super(KLDistiller, self).__init__()
        self.T = T

    def forward(self, s_y, t_y):
        # --------------------------------------------
        # Shape of s_y : (bs, n_classes)
        # Shape of t_y : (bs, n_classes)
        # --------------------------------------------
        s_p = F.log_softmax(s_y / self.T, dim=1)
        t_p = F.softmax(t_y / self.T, dim=1)
        loss = F.kl_div(s_p, t_p, size_average=False) * (self.T * self.T) / s_y.shape[0]
        return loss
