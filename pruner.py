import numpy as np
import torch
from torch.nn.modules.module import Module

from util import z_score_v2, min_max_scalar


class FiltersPruningModule(Module):
    def __init__(self):
        super(FiltersPruningModule, self).__init__()
        self.conv_mask = dict()

    @staticmethod
    def _get_prune_indices(conv_module, prune_rates, mode='filter-norm'):
        f_w = conv_module.weight.data.cpu().numpy()  # The weight of filters
        f_g = conv_module.weight.grad.cpu().numpy()  # The gradient of filters
        if 'filter-norm' in mode:
            sum_of_objs = np.sum(np.abs(f_w.reshape(f_w.shape[0], -1)), 1)
            num_of_objs = f_w.shape[0]
        elif 'channel-norm' in mode:
            perm_f_w = np.transpose(f_w, (1, 0, 2, 3))  # (fn, cn, kh, kw) => (cn, fn, kh, kw)
            sum_of_objs = np.sum(np.abs(perm_f_w.reshape(perm_f_w.shape[0], -1)), 1)
            num_of_objs = f_w.shape[1]
        elif 'filter-gm' in mode:  # Geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            sum_of_objs = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            num_of_objs = f_w.shape[0]
        elif 'filter-ggm' in mode:  # Combine gradient-base and geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            _sum_of_dists = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            _sum_of_grads = np.sum(np.abs(f_g.reshape(f_g.shape[0], -1)), 1)
            sum_of_objs = _sum_of_dists * _sum_of_grads
            num_of_objs = f_w.shape[0]
        elif 'filter-ggm2' in mode:  # Combine gradient-base and geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            _sum_of_dists = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            _sum_of_grads = np.sum(np.abs(f_g.reshape(f_g.shape[0], -1)), 1)
            sum_of_objs = _sum_of_dists + _sum_of_grads
            num_of_objs = f_w.shape[0]
        elif 'filter-ggm3' in mode:  # Combine norm-gradient-base and norm geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            _sum_of_weights = np.sum(np.abs(f_w.reshape(f_w.shape[0], -1)), 1)
            _sum_of_dists = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            _sum_of_grads = np.sum(np.abs(f_g.reshape(f_g.shape[0], -1)), 1)
            sum_of_objs = _sum_of_dists * _sum_of_grads * _sum_of_weights
            num_of_objs = f_w.shape[0]
        elif 'filter-nggm' in mode:  # Combine norm-gradient-base and norm geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            _sum_of_dists = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            _sum_of_grads = np.sum(np.abs(f_g.reshape(f_g.shape[0], -1)), 1)
            sum_of_objs = min_max_scalar(_sum_of_dists) + min_max_scalar(_sum_of_grads)
            num_of_objs = f_w.shape[0]
        elif 'filter-ga' in mode:  # Combine gradient-base and activation-base
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            flat_f_g = f_g.reshape(f_g.shape[0], -1)
            sum_of_objs = np.sum(np.abs(flat_f_w) * np.abs(flat_f_g), 1)
            num_of_objs = f_w.shape[0]
        elif 'filter-nga' in mode:  # Combine norm-gradient-base and norm-activation-base
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            flat_f_g = f_g.reshape(f_g.shape[0], -1)
            sum_of_objs = np.sum(min_max_scalar(np.abs(flat_f_w)) + min_max_scalar(np.abs(flat_f_g)), 1)
            num_of_objs = f_w.shape[0]
        elif 'filter-nga2' in mode:  # Combine norm-gradient-base and norm-activation-base
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            flat_f_g = f_g.reshape(f_g.shape[0], -1)
            sum_of_objs = min_max_scalar(np.sum(np.abs(flat_f_w), 1)) + min_max_scalar(np.sum(np.abs(flat_f_g), 1))
            num_of_objs = f_w.shape[0]
        else:
            raise NameError
        idx_of_objs = np.argsort(sum_of_objs)
        prune_num_of_objs = round(num_of_objs * prune_rates)
        prune_idx_of_objs = np.sort(idx_of_objs[:prune_num_of_objs])
        print(prune_idx_of_objs)
        return prune_idx_of_objs

    @staticmethod
    def _prune_by_indices(module, indices, dim=0):
        weight = module.weight.data
        bias = module.bias.data
        if dim == 0:
            weight[indices] = 0.0
            bias[indices] = 0.0
        elif dim == 1:
            weight[:, indices] = 0.0

    def _prune_by_percentile(self, layers, q=5.0):
        for name, module in self.named_modules():
            if name in layers:
                tensor = module.weight.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), q)
                print(f'Pruning {name} with threshold : {percentile_value}')
                self._prune_by_threshold(module, percentile_value)

    def _init_conv_mask(self, name, module):
        self.conv_mask[name] = np.ones(module.weight.data.shape)

    def _set_conv_mask(self, name, prune_indices, dim=0):
        mask_arr = self.conv_mask[name]
        if dim == 0:
            mask_arr[prune_indices] = 0
        elif dim == 1:
            mask_arr[:, prune_indices] = 0

    def _prune_filters_and_channels(self, prune_rates, mode='hard-filter-norm'):
        i = 0
        dim = 0
        prune_indices = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._init_conv_mask(name, module)
                if dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=dim)
                    self._set_conv_mask(name, prune_indices, dim=dim)
                    dim = 0
                prune_indices = self._get_prune_indices(module, prune_rates[i], mode=mode)
                self._prune_by_indices(module, prune_indices, dim=dim)
                self._set_conv_mask(name, prune_indices, dim=dim)
                dim = 1
                i += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=0)

    def prune(self, mode, ideal_prune_rates):
        if 'percentile' in mode:
            self._prune_by_percentile(ideal_prune_rates)
        elif 'filter' in mode:
            act_prune_rates = self.get_conv_act_prune_rates(ideal_prune_rates)
            self._prune_filters_and_channels(act_prune_rates, mode=mode)

    def get_conv_act_prune_rates(self, ideal_prune_rates, verbose=False):
        """ Suppose the model prunes some filters (filters, :, :, :). """
        i = 0
        n_prune_filters = None
        prune_rates = list()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                n_filters, n_channels = module.weight.shape[0:2]
                # ---------------------------------------------
                # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
                # prune
                # ---------------------------------------------
                # Suppose Conv Shape: (fn, cn, kh, kw)
                # Then:
                #   (1 - ideal_prune_rate) = ((fn * (1 - ideal_f_prune_rate)) * (cn - n_prune_filters) * kh * kw) / (fn
                #   * cn * kh * kw)
                #
                # "n_prune_filters" is the number of prune filters last layer
                # ---------------------------------------------
                if i == 0:
                    ideal_f_prune_rate = ideal_prune_rates[i]
                else:
                    ideal_f_prune_rate = (
                        1 - ((1 - ideal_prune_rates[i]) * (n_channels / (n_channels - n_prune_filters)))
                    )
                n_prune_filters = round(n_filters * ideal_f_prune_rate)
                act_f_prune_rate = n_prune_filters / n_filters
                f_bias = abs(act_f_prune_rate - ideal_f_prune_rate)
                prune_rates.append(act_f_prune_rate)
                i += 1
                if verbose:
                    print(f'{name:6} | original filter nums: {n_filters:4} | prune filter nums: '
                          f'{n_prune_filters:4} | target filter prune rate: {ideal_f_prune_rate * 100.:.2f}'
                          f'% | actual filter prune rate : {act_f_prune_rate * 100.:.2f}% | filter bias: '
                          f'{f_bias * 100.:.2f}%')
        return prune_rates






