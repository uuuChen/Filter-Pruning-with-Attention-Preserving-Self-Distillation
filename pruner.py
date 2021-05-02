import numpy as np
import torch
from torch.nn.modules.module import Module

from util import z_score_v2, min_max_scalar


class FiltersPruningModule(Module):
    def __init__(self):
        super(FiltersPruningModule, self).__init__()
        self.conv_mask = dict()

    @staticmethod
    def _get_prune_indices(conv_module, prune_rate, mode='filter-norm'):
        def get_gm_dists(weights):
            return np.array([np.sum(np.power(weights - weight, 2)) for weight in weights])

        def get_l1_scores(weights_2d):
            return np.sum(np.abs(weights_2d), axis=1)

        f_weights = conv_module.weight.data.cpu().numpy()  # The weight of filters
        f_grads = conv_module.weight.grad.data.cpu().numpy()  # The gradient of filters
        f_nums = f_weights.shape[0]
        flat_f_weights = f_weights.reshape(f_nums, -1)
        flat_f_grads = f_grads.reshape(f_nums, -1)
        if 'filter-norm' in mode:
            f_scores = get_l1_scores(flat_f_weights)
        elif 'filter-gm' in mode:  # Geometric-median
            f_scores = get_gm_dists(flat_f_weights)
        elif 'filter-ggm' in mode:  # Combine gradient-base and geometric-median
            f_scores = get_gm_dists(flat_f_weights) * get_l1_scores(flat_f_grads)
        elif 'filter-ggm2' in mode:  # Combine gradient-base and geometric-median
            f_scores = get_gm_dists(flat_f_weights) + get_l1_scores(flat_f_grads)
        elif 'filter-ggm3' in mode:  # Combine norm-gradient-base and norm-geometric-median
            f_scores = get_gm_dists(flat_f_weights) * get_l1_scores(flat_f_grads) * get_l1_scores(flat_f_weights)
        elif 'filter-nggm' in mode:  # Combine norm-gradient-base and norm-geometric-median
            f_scores = min_max_scalar(get_gm_dists(flat_f_weights)) + min_max_scalar(get_l1_scores(flat_f_grads))
        elif 'filter-ga' in mode:  # Combine gradient-base and activation-base
            f_scores = np.sum(np.abs(flat_f_weights) * np.abs(flat_f_grads), 1)
        elif 'filter-nga' in mode:  # Combine norm-gradient-base and norm-activation-base
            f_scores = np.sum(min_max_scalar(np.abs(flat_f_weights)) + min_max_scalar(np.abs(flat_f_grads)), 1)
        elif 'filter-nga2' in mode:  # Combine norm-gradient-base and norm-activation-base
            f_scores = min_max_scalar(get_l1_scores(flat_f_weights)) + min_max_scalar(get_l1_scores(flat_f_grads))
        else:
            raise NameError
        rank_f_indices = np.argsort(f_scores)
        prune_f_nums = round(f_nums * prune_rate)
        prune_f_indices = np.sort(rank_f_indices[:prune_f_nums])
        print(prune_f_indices)
        return prune_f_indices

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






