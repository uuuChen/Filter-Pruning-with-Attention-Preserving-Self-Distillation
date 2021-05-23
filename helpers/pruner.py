import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from helpers.utils import min_max_scalar


class FiltersPruner(object):
    def __init__(self, model, optimizer, train_loader, device, logger, use_PFEC=False):
        super(FiltersPruner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.logger = logger
        self.use_PFEC = use_PFEC

        self.cross_entropy = nn.CrossEntropyLoss()
        self.conv_mask = dict()

    def _get_prune_indices(self, conv_name, conv_module, prune_rate, mode='filter-a'):
        def get_gm_dists(weights):
            return np.array([np.sum(np.power(weights - weight, 2)) for weight in weights])

        def get_l1_scores(weights_2d):
            return np.sum(np.abs(weights_2d), axis=1)

        f_weights = conv_module.weight.data.cpu().numpy()  # The weight of filters
        f_nums = f_weights.shape[0]
        flat_f_weights = f_weights.reshape(f_nums, -1)
        flat_f_grads = None
        if '-g-' in mode:
            f_grads = conv_module.weight.grad.data.cpu().numpy()  # The gradient of filters
            flat_f_grads = f_grads.reshape(f_nums, -1)

        if 'filter-a' in mode:
            f_scores = get_l1_scores(flat_f_weights)
        elif 'filter-gm' in mode:  # Geometric-median
            f_scores = get_gm_dists(flat_f_weights)
        elif 'filter-g-gm-1' in mode:  # Combine gradient-base and geometric-median
            f_scores = get_gm_dists(flat_f_weights) * get_l1_scores(flat_f_grads)
        elif 'filter-g-gm-2' in mode:  # Combine gradient-base and geometric-median
            f_scores = get_gm_dists(flat_f_weights) + get_l1_scores(flat_f_grads)
        elif 'filter-g-gm-3' in mode:  # Combine norm-gradient-base and norm-geometric-median
            f_scores = get_gm_dists(flat_f_weights) * get_l1_scores(flat_f_grads) * get_l1_scores(flat_f_weights)
        elif 'filter-n-g-gm-1' in mode:  # Combine norm-gradient-base and norm-geometric-median
            f_scores = min_max_scalar(get_gm_dists(flat_f_weights)) + min_max_scalar(get_l1_scores(flat_f_grads))
        elif 'filter-n-g-gm-2' in mode:
            f_scores = (min_max_scalar(get_gm_dists(flat_f_weights)) + min_max_scalar(get_l1_scores(flat_f_grads)) +
                        min_max_scalar(get_l1_scores(flat_f_weights)))
        elif 'filter-n-g-gm-3' in mode:  # Combine norm-gradient-base and norm-geometric-median
            f_scores = min_max_scalar(get_gm_dists(flat_f_weights)) * min_max_scalar(get_l1_scores(flat_f_grads))
        elif 'filter-g-a' in mode:  # Combine gradient-base and activation-base
            f_scores = np.sum(np.abs(flat_f_weights) * np.abs(flat_f_grads), 1)
        elif 'filter-n-g-a' in mode:  # Combine norm-gradient-base and norm-activation-base
            f_scores = np.sum(min_max_scalar(np.abs(flat_f_weights)) + min_max_scalar(np.abs(flat_f_grads)), 1)
        elif 'filter-n-g-a-2' in mode:  # Combine norm-gradient-base and norm-activation-base
            f_scores = min_max_scalar(get_l1_scores(flat_f_weights)) + min_max_scalar(get_l1_scores(flat_f_grads))
        else:
            raise NameError
        rank_f_indices = np.argsort(f_scores)
        prune_f_nums = int(round(f_nums * (1.0 - prune_rate)))
        prune_f_indices = np.sort(rank_f_indices[:prune_f_nums])
        self.logger.log(f'{conv_name:10} Prune-F-Indices : {prune_f_indices}')
        return prune_f_indices

    @staticmethod
    def _prune_by_indices(module, indices, dim=0):
        weight = module.weight.data
        bias = None
        is_bias_exist = True if module.bias is not None else False
        if is_bias_exist:
            bias = module.bias.data
        if dim == 0:
            weight[indices] = 0.0
            if is_bias_exist:
                bias[indices] = 0.0
        elif dim == 1:
            weight[:, indices] = 0.0

    @staticmethod
    def _prune_by_threshold(module, threshold):
        device = module.weight.device
        tensor = module.weight.data.cpu().numpy()
        mask = np.where(abs(tensor) < threshold, 0, 1)
        module.weight.data = torch.from_numpy(tensor * mask).to(device)

    def _check_prune_rates(self, prune_rates):
        i = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                i += 1
        if not (len(prune_rates) == 1 or len(prune_rates) == i):
            raise ValueError(f"Prune Rates Nums ({len(prune_rates)}) / Layers Nums ({i}) Mismatch")
        if len(prune_rates) == 1:  # Length 1 => Length i
            prune_rates *= i
        return prune_rates

    def _prune_by_percentile(self, prune_rates):
        i = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                tensor = module.weight.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), prune_rates[i])
                print(f'Pruning {name} with threshold : {percentile_value}')
                self._prune_by_threshold(module, percentile_value)
                i += 1

    def _init_conv_mask(self, name, module):
        self.conv_mask[name] = np.ones(module.weight.data.shape)

    def _set_conv_mask(self, name, prune_indices, dim=0):
        mask_arr = self.conv_mask[name]
        if dim == 0:
            mask_arr[prune_indices] = 0
        elif dim == 1:
            mask_arr[:, prune_indices] = 0

    # def _set_epoch_acc_weights_grad(self):
    #     params = list(self.model.parameters())
    #     acc_grads = None
    #     iter_bar = tqdm(self.train_loader)
    #     for i, batch in enumerate(iter_bar):
    #         input_var, target_var = [t.to(self.device) for t in batch]
    #         self.optimizer.zero_grad()
    #         output_var = self.model(input_var)
    #         loss = self.cross_entropy(output_var, target_var)
    #         loss.backward()
    #         if acc_grads is None:
    #             acc_grads = np.array(
    #                 [torch.zeros(p.grad.shape, dtype=torch.float64).to(self.device) for p in params],
    #                 dtype=object
    #             )
    #         acc_grads += np.array([p.grad for p in params], dtype=object)
    #     acc_grads /= len(iter_bar)
    #     for p, acc_grad in zip(params, acc_grads):
    #         p.grad.data = acc_grad

    def _set_epoch_acc_weights_grad(self):
        params = list(self.model.parameters())
        train_iter = iter(self.train_loader)
        input_var, target_var = [t.to(self.device) for t in next(train_iter)]
        for i, batch in enumerate(train_iter):
            _input_var, _target_var = [t.to(self.device) for t in batch]
            input_var = torch.cat((input_var, _input_var), dim=0)
            target_var = torch.cat((target_var, _target_var), dim=0)
            if i == 3:
                break
        self.optimizer.zero_grad()
        output_var = self.model(input_var)
        loss = self.cross_entropy(output_var, target_var)
        loss.backward()
        acc_grads = np.array([p.grad for p in params], dtype=object)
        for p, acc_grad in zip(params, acc_grads):
            p.grad.data = acc_grad

    def _prune_filters_and_channels(self, prune_rates, mode='hard-filter-norm'):
        i = 0
        dim = 0
        prune_indices = None
        use_grad = '-g-' in mode
        if use_grad:
            self._set_epoch_acc_weights_grad()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._init_conv_mask(name, module)
                if dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=dim)
                    self._set_conv_mask(name, prune_indices, dim=dim)
                    dim = 0
                prune_indices = self._get_prune_indices(name, module, prune_rates[i], mode=mode)
                self._prune_by_indices(module, prune_indices, dim=dim)
                self._set_conv_mask(name, prune_indices, dim=dim)
                dim = 1
                i += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=0)
        if use_grad:
            self.optimizer.zero_grad()

    def _get_conv_act_prune_rates(self, ideal_prune_rates, verbose=False):
        """
        # Suppose the model prunes some filters (filters, :, :, :).
        # ---------------------------------------------
        # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
        # prune
        # ---------------------------------------------
        # Suppose Conv Shape: (fn, cn, kh, kw)
        # Then:
        #   ideal_prune_rate = ((fn * ideal_f_prune_rate) * (cn - n_prune_filters) * kh * kw) / (fn * cn * kh
        #   * kw)
        $ Then:
        #   ideal_f_prune_rate = ideal_prune_rate * (n_channels / (n_channels - n_prune_filters))
        #
        # "n_prune_filters" is the number of prune filters last layer
        # ---------------------------------------------
        """
        i = 0
        n_prune_filters = None
        prune_rates = list()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                n_filters, n_channels = module.weight.shape[0:2]
                if i == 0:
                    ideal_f_prune_rate = ideal_prune_rates[i]
                else:
                    ideal_f_prune_rate = ideal_prune_rates[i] * (n_channels / (n_channels - n_prune_filters))
                n_prune_filters = round(n_filters * (1.0 - ideal_f_prune_rate))
                act_f_prune_rate = np.clip(1.0 - (n_prune_filters / n_filters), 0.0, 1.0)
                f_bias = abs(act_f_prune_rate - ideal_f_prune_rate)
                prune_rates.append(act_f_prune_rate)
                i += 1
                if verbose:
                    print(f'{name:6} | original filter nums: {n_filters:4} | prune filter nums: '
                          f'{n_prune_filters:4} | target filter prune rate: {ideal_f_prune_rate * 100.:.2f}'
                          f'% | actual filter prune rate : {act_f_prune_rate * 100.:.2f}% | filter bias: '
                          f'{f_bias * 100.:.2f}%')
        return prune_rates

    def prune(self, mode, ideal_prune_rates):
        ideal_prune_rates = self._check_prune_rates(ideal_prune_rates)
        if 'percentile' in mode:
            self._prune_by_percentile(ideal_prune_rates)
        elif 'filter' in mode:
            prune_rates = ideal_prune_rates
            if self.use_PFEC:
                prune_rates = self._get_conv_act_prune_rates(ideal_prune_rates)
            self._prune_filters_and_channels(prune_rates, mode=mode)







