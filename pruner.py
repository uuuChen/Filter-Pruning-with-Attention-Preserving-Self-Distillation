import numpy as np
import torch
from torch.nn.modules.module import Module

from util import z_score_v2


class FilterPruningModule(Module):
    def __init__(self):
        super(FilterPruningModule, self).__init__()
        self.name_to_prune_indices = dict()
        self.name_to_left_indices = dict()

    @staticmethod
    def _get_prune_indices(conv_module, prune_rates, mode='filter-norm'):
        f_w = conv_module.weight.data.cpu().numpy()  # The weight of filters
        f_g = conv_module.weight.grad.cpu().numpy()  # The gradient of filters
        sum_of_objs = None
        num_of_objs = None
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
        elif 'filter-nggm' in mode:  # Combine norm-gradient-base and norm geometric-median
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            _sum_of_dists = np.array([np.sum(np.power(flat_f_w - per_f_w, 2)) for per_f_w in flat_f_w])
            _sum_of_grads = np.sum(np.abs(f_g.reshape(f_g.shape[0], -1)), 1)
            sum_of_objs = z_score_v2(_sum_of_dists) * z_score_v2(_sum_of_grads)
            num_of_objs = f_w.shape[0]
        elif 'filter-ga' in mode:  # Combine gradient-base and activation-base
            flat_f_w = f_w.reshape(f_w.shape[0], -1)
            grad_flat_arr = f_g.reshape(f_g.shape[0], -1)
            sum_of_objs = np.sum(np.abs(flat_f_w) * np.abs(grad_flat_arr), 1)
            num_of_objs = f_w.shape[0]
        elif 'filter-nga' in mode:  # Combine norm-gradient-base and norm-activation-base
            filters_flat_arr = f_w.reshape(f_w.shape[0], -1)
            grad_flat_arr = f_g.reshape(f_g.shape[0], -1)
            sum_of_objs = np.sum(z_score_v2(np.abs(filters_flat_arr)) * z_score_v2(np.abs(grad_flat_arr)), 1)
            num_of_objs = f_w.shape[0]
        object_indices = np.argsort(sum_of_objs)
        pruned_object_nums = round(num_of_objs * prune_rates)
        pruned_indices = np.sort(object_indices[:pruned_object_nums])
        print(pruned_indices)
        return pruned_indices

    @staticmethod
    def _prune_by_indices(module, dim, indices):
        if dim == 0:
            if len(module.weight.size()) == 4:  # conv layer etc.
                module.weight.data[indices, :, :, :] = 0.0
            elif len(module.weight.size()) == 1:  # conv_bn layer etc.
                module.weight.data[indices] = 0.0
            module.bias.data[indices] = 0.0
        elif dim == 1:
            module.weight.data[:, indices, :, :] = 0.0  # only happened to conv layer, so its dimension is 4

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
        dim = 0
        conv_idx = 0
        prune_indices = None
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if 'filter' in mode:
                    if dim == 1:
                        self._prune_by_indices(module, dim, prune_indices)
                        dim ^= 1
                    prune_indices = self._get_prune_indices(module, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indices(module, dim, prune_indices)
                    dim ^= 1
                elif 'channel' in mode:
                    dim = 1
                    prune_indices = self._get_prune_indices(module, prune_rates[conv_idx], mode=mode)
                    self._prune_by_indices(module, dim, prune_indices)
                conv_idx += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indices(module, 0, prune_indices)

    def prune(self, mode, prune_rates):
        if 'percentile' in mode:
            self._prune_by_percentile(prune_rates)
        elif 'filter' in mode:
            prune_rates = self.get_filters_prune_rates(prune_rates)
            self._prune_filters_or_channels(prune_rates, mode=mode)
            self.set_indices_dicts_after_pruning()

    def get_filters_prune_rates(self, ideal_prune_rates, mode='filter', print_log=False):
        """ Suppose the model prunes some filters (filters, :, :, :) or channels (:, channels, :, :). """
        conv_idx = 0
        prune_filter_nums = None
        new_pruning_rates = list()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_shape = module.weight.shape
                filter_nums = conv_shape[0]
                channel_nums = conv_shape[1]

                ideal_filter_prune_rate = ideal_channel_prune_rate = ideal_prune_rates[conv_idx]

                # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
                # prune
                # Suppose Conv Shape: (fn, cn, kh, kw)
                # (1 - prune_rates) = ((fn * (1 - new_prune_rate)) * (cn - Prune_filter_nums) * kh * kw) / (fn * cn * kh
                # * kw)
                if conv_idx != 0:
                    ideal_filter_prune_rate = (
                        1 - ((1 - ideal_filter_prune_rate) * (channel_nums / (channel_nums - prune_filter_nums)))
                    )

                prune_filter_nums = round(filter_nums * ideal_filter_prune_rate)
                actual_filter_prune_rate = prune_filter_nums / filter_nums
                filter_bias = abs(actual_filter_prune_rate - ideal_filter_prune_rate)

                prune_channel_nums = round(channel_nums * ideal_channel_prune_rate)
                actual_channel_prune_rate = prune_channel_nums / channel_nums
                channel_bias = abs(actual_channel_prune_rate - ideal_channel_prune_rate)

                if mode == 'filter':
                    if print_log:
                        print(f'{name:6} | original filter nums: {filter_nums:4} | prune filter nums: '
                              f'{prune_filter_nums:4} | target filter prune rate: {ideal_filter_prune_rate * 100.:.2f}'
                              f'% | actual filter prune rate : {actual_filter_prune_rate * 100.:.2f}% | filter bias: '
                              f'{filter_bias * 100.:.2f}%')
                    new_pruning_rates.append(actual_filter_prune_rate)
                elif mode == 'channel':
                    if print_log:
                        print(f'{name:6} | original channel nums: {channel_nums:4} | prune channel nums: '
                              f'{prune_channel_nums:4} | target channel prune rate: '
                              f'{ideal_channel_prune_rate * 100.:.2f}% | actual channel prune rate: '
                              f'{actual_channel_prune_rate * 100.:.2f}% | channel bias: {channel_bias * 100.:.2f}%')
                    new_pruning_rates.append(actual_channel_prune_rate)

                conv_idx += 1

        return new_pruning_rates

    def set_indices_dicts_after_pruning(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_arr = module.weight.data.cpu().numpy()
                perm_conv_arr = np.transpose(conv_arr, (1, 0, 2, 3))  # (fn, cn, kh, kw) => (cn, fn, kh, kw)

                prune_filters_indices = np.where(
                    np.sum(conv_arr.reshape(conv_arr.shape[0], -1), axis=1) == 0
                )[0]
                prune_channels_indices = np.where(
                    np.sum(perm_conv_arr.reshape(perm_conv_arr.shape[0], -1), axis=1) == 0
                )[0]

                left_filters_indices = list(set(range(conv_arr.shape[0])).difference(prune_filters_indices))
                left_channels_indices = list(set(range(perm_conv_arr.shape[0])).difference(prune_channels_indices))

                self.name_to_prune_indices[name] = (prune_filters_indices, prune_channels_indices)
                self.name_to_left_indices[name] = (left_filters_indices, left_channels_indices)






