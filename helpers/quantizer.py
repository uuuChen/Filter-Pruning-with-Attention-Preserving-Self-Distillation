import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn


class PostQuantizer:
    def __init__(self, quan_mode, device='cuda'):
        self.device = device
        self.do_c_quan = 'conv' in quan_mode
        self.do_f_quan = 'fc' in quan_mode
        self.quan_dict = dict()

    def get_quan_dict(self):
        return self.quan_dict

    def quantize(self, model, bits):
        assert isinstance(bits, int) or isinstance(bits, dict)
        for name, module in model.named_modules():
            if not (isinstance(module, nn.Conv2d) and not self.do_f_quan or
                    isinstance(module, nn.Linear) and not self.do_c_quan):
                continue
            ori_w = module.weight.data.cpu().numpy()
            n_uni_w = len(np.unique(ori_w))
            quan_range = np.power(2, bits) if isinstance(bits, int) else np.power(2, bits[name])
            if quan_range >= n_uni_w:
                continue

            print(f'{name:20} | {str(ori_w.shape):35} | => quantize to {quan_range} indices')
            left_w = ori_w[ori_w != 0].reshape(-1, 1)
            space = np.linspace(np.min(left_w), np.max(left_w), num=quan_range).reshape(-1, 1)
            kmeans = KMeans(n_clusters=len(space), init=space, n_init=1, algorithm="full")
            kmeans.fit(left_w)

            left_ind = np.where(ori_w != 0)
            quan_w = np.zeros(ori_w.shape)
            quan_w[left_ind] = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            module.weight.data = torch.from_numpy(quan_w).float().to(self.device)

            quan_labels = np.zeros(ori_w.shape)
            quan_labels[left_ind] = kmeans.labels_

            self.quan_dict[name] = quan_labels



