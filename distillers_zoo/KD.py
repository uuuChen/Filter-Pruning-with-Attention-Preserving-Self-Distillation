import torch.nn as nn
import torch.nn.functional as F


class KLDistiller(nn.Module):
    def __init__(self, T):
        super(KLDistiller, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T * self.T) / y_s.shape[0]
        return loss
