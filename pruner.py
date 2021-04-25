import numpy as np
import torch
from torch.nn.modules.module import Module

from util import z_score_v2, min_max_scalar


class FiltersPruningModule(Module):
    def __init__(self):
        super(FiltersPruningModule, self).__init__()

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
        weight_dim = len(module.weight.size())
        if dim == 0:
            if weight_dim == 4:  # conv layer etc.
                weight[indices, :, :, :] = 0.0
            elif weight_dim == 1:  # conv_bn layer etc.
                weight[indices] = 0.0
            bias[indices] = 0.0
        elif dim == 1:
            weight[:, indices, :, :] = 0.0  # only happened to conv layer, so its dimension is 4

    def _prune_by_percentile(self, layers, q=5.0):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        for name, module in self.named_modules():
            if name in layers:
                tensor = module.weight.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), q)
                print(f'Pruning {name} with threshold : {percentile_value}')
                self._prune_by_threshold(module, percentile_value)

    def _prune_filters_or_channels(self, prune_rates, mode='filter-norm'):
        i = 0
        dim = 0
        prune_indices = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if 'filter' in mode:
                    if dim == 1:
                        self._prune_by_indices(module, prune_indices, dim=dim)
                        dim ^= 1
                    prune_indices = self._get_prune_indices(module, prune_rates[i], mode=mode)
                    self._prune_by_indices(module, prune_indices, dim=dim)
                    dim ^= 1
                elif 'channel' in mode:
                    dim = 1
                    prune_indices = self._get_prune_indices(module, prune_rates[i], mode=mode)
                    self._prune_by_indices(module, prune_indices, dim=dim)
                i += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=0)

    def prune(self, mode, prune_rates):
        if 'percentile' in mode:
            self._prune_by_percentile(prune_rates)
        elif 'filter' in mode:
            conv_prune_rates = self.get_conv_prune_rates(prune_rates)
            self._prune_filters_or_channels(conv_prune_rates, mode=mode)

    def get_conv_prune_rates(self, ideal_prune_rates, mode='filter', verbose=False):
        """ Suppose the model prunes some filters (filters, :, :, :) or channels (:, channels, :, :). """
        i = 0
        n_act_prune_f = None  # Actual nums of prune filters
        prune_rates = list()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                n_f, n_c = module.weight.shape[0:2]
                # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
                # prune
                # Suppose Conv Shape: (fn, cn, kh, kw)
                # (1 - ideal_prune_rate) = ((fn * (1 - ideal_f_prune_rate)) * (cn - n_act_prune_f) * kh * kw) / (fn
                # * cn * kh * kw)
                if i == 0:
                    ideal_f_prune_rate = ideal_prune_rates[i]
                else:
                    ideal_f_prune_rate = 1 - ((1 - ideal_prune_rates[i]) * (n_c / (n_c - n_act_prune_f)))
                ideal_c_prune_rate = ideal_prune_rates[i]

                n_act_prune_f = round(n_f * ideal_f_prune_rate)
                act_f_prune_rate = n_act_prune_f / n_f
                f_bias = abs(act_f_prune_rate - ideal_f_prune_rate)

                n_act_prune_c = round(n_c * ideal_c_prune_rate)
                act_c_prune_rate = n_act_prune_c / n_c
                c_bias = abs(act_c_prune_rate - ideal_c_prune_rate)

                if mode == 'filter':
                    if verbose:
                        print(f'{name:6} | original filter nums: {n_f:4} | prune filter nums: '
                              f'{n_act_prune_f:4} | target filter prune rate: {ideal_f_prune_rate * 100.:.2f}'
                              f'% | actual filter prune rate : {act_f_prune_rate * 100.:.2f}% | filter bias: '
                              f'{f_bias * 100.:.2f}%')
                    prune_rates.append(act_f_prune_rate)
                elif mode == 'channel':
                    if verbose:
                        print(f'{name:6} | original channel nums: {n_c:4} | prune channel nums: '
                              f'{n_act_prune_c:4} | target channel prune rate: '
                              f'{ideal_c_prune_rate * 100.:.2f}% | actual channel prune rate: '
                              f'{act_c_prune_rate * 100.:.2f}% | channel bias: {c_bias * 100.:.2f}%')
                    prune_rates.append(act_c_prune_rate)
                i += 1
        return prune_rates






