import numpy as np

import torch
import torch.nn as nn

from helpers.utils import min_max_scalar


class FiltersPruner(object):
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 logger,
                 samp_batches=None,
                 device='cuda',
                 use_actPR=False,
                 use_greedy=False):
        super(FiltersPruner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.samp_batches = samp_batches
        self.logger = logger
        self.use_actPR = use_actPR
        self.use_greedy = use_greedy

        self.cross_entropy = nn.CrossEntropyLoss()
        self.conv_mask = dict()
        self.use_grad = False

    def _get_prune_indices(self, name, module, prune_rate, mode='filter-a'):
        f_w = module.weight.data.cpu().numpy()  # The weight of filters
        n_f = f_w.shape[0]
        f_w = f_w.reshape(n_f, -1)

        f_g = None
        if self._get_use_grad():
            f_g = module.weight.grad.data.cpu().numpy()  # The gradient of filters
            f_g = f_g.reshape(n_f, -1)

        # ------------------------------------
        # In mode:
        #    -g- : Combine gradients
        #    -n- : Combine min-max-scalar
        #    -a  : Use activation-base
        #    -gm : Use geometric-median-base
        # ------------------------------------

        def get_gm_dists(arr):
            # --------------------------------------------
            # Shape of arr : (n_f, n_c * h * w)
            # --------------------------------------------
            return np.array([np.sum(np.power(ele - arr, 2)) for ele in arr])  # (n_f,)

        def get_l1_scores(arr):
            # --------------------------------------------
            # Shape of arr: (n_f, n_c * h * w)
            # --------------------------------------------
            return np.sum(np.abs(arr), axis=1)  # (n_f,)

        if 'filter-a' in mode:
            f_scores = get_l1_scores(f_w)
        elif 'filter-g-a' in mode:
            f_scores = np.sum(np.abs(f_w) * np.abs(f_g), axis=1)
        elif 'filter-n-g-a' in mode:
            f_scores = np.sum(min_max_scalar(np.abs(f_w)) + min_max_scalar(np.abs(f_g)), axis=1)
        elif 'filter-n-g-a-2' in mode:
            f_scores = min_max_scalar(get_l1_scores(f_w)) + min_max_scalar(get_l1_scores(f_g))
        elif 'filter-gm' in mode:
            f_scores = get_gm_dists(f_w)
        elif 'filter-g-gm-1' in mode:
            f_scores = get_gm_dists(f_w) * get_l1_scores(f_g)
        elif 'filter-g-gm-2' in mode:
            f_scores = get_gm_dists(f_w) + get_l1_scores(f_g)
        elif 'filter-g-gm-3' in mode:
            f_scores = get_gm_dists(f_w) * get_l1_scores(f_g) * get_l1_scores(f_w)
        elif 'filter-n-g-gm-1' in mode:
            f_scores = min_max_scalar(get_gm_dists(f_w)) + min_max_scalar(get_l1_scores(f_g))
        elif 'filter-n-g-gm-2' in mode:
            f_scores = (min_max_scalar(get_gm_dists(f_w)) + min_max_scalar(get_l1_scores(f_g)) +
                        min_max_scalar(get_l1_scores(f_w)))
        elif 'filter-n-g-gm-3' in mode:
            f_scores = min_max_scalar(get_gm_dists(f_w)) * min_max_scalar(get_l1_scores(f_g))
        else:
            raise NameError
        rank_indices = np.argsort(f_scores)
        n_prune = int(round(n_f * (1.0 - prune_rate)))
        prune_indices = np.sort(rank_indices[:n_prune])
        self.logger.log(f'{name:10} Prune-F-Indices : {prune_indices}')
        return prune_indices

    @staticmethod
    def _prune_by_threshold(module, threshold):
        weight = module.weight
        tensor = weight.data.cpu().numpy()
        mask = np.where(abs(tensor) < threshold, 0, 1)
        weight.data = torch.from_numpy(tensor * mask).to(weight.device)

    @staticmethod
    def _prune_by_indices(module, indices, dim=0, prune_weight=True, prune_bias=True, prune_grad=True):
        l = list()
        weight = module.weight
        bias = module.bias
        grad = module.weight.grad
        if prune_weight:
            l.append(weight)
        if prune_bias and bias is not None:
            l.append(bias)
        if prune_grad and grad is not None:
            l.append(grad)
        if dim == 0:
            for t in l:
                t.data[indices] = 0.0
        elif dim == 1:
            for t in l:
                t.data[:, indices] = 0.0

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
                alive = tensor[np.nonzero(tensor)]  # Flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), prune_rates[i])
                print(f'Pruning {name} with threshold : {percentile_value}')
                self._prune_by_threshold(module, percentile_value)
                i += 1

    def _init_conv_mask(self, name, module):
        self.conv_mask[name] = torch.ones(module.weight.data.shape).to(self.device)

    def _set_conv_mask(self, name, prune_indices, dim=0):
        mask_arr = self.conv_mask[name]
        if dim == 0:
            mask_arr[prune_indices] = 0
        elif dim == 1:
            mask_arr[:, prune_indices] = 0

    def _get_use_grad(self):
        return self.use_grad

    def _set_use_grad(self, val=False):
        self.use_grad = val

    def _set_batches_weight_grad(self):
        device = 'cpu'
        self.model = self.model.to(device)
        params = list(self.model.parameters())
        train_iter = iter(self.train_loader)
        input, target = [t.to(device) for t in next(train_iter)]
        for i, batch in enumerate(train_iter, start=1):
            if i == self.samp_batches:
                break
            inp, tar = [t.to(device) for t in batch]
            input = torch.cat((input, inp), dim=0)
            target = torch.cat((target, tar), dim=0)
        self.optimizer.zero_grad()
        logit = self.model(input)
        loss = self.cross_entropy(logit, target)
        loss.backward()
        grads = np.array([p.grad for p in params], dtype=object)
        for p, grad in zip(params, grads):
            p.grad.data = grad
        self.model = self.model.to(self.device)

    def _prune_filters_and_channels(self, prune_rates, mode='filter-norm'):
        """
        # We imitate FPGM (CVPR - 2019 oral) and don’t have prune channels
        # Github: https://github.com/he-y/filter-pruning-geometric-median/blob/master/pruning_cifar10.py
        """
        i = 0
        dim = 0
        prune_indices = prune_indices_ = None
        if self._get_use_grad():
            self._set_batches_weight_grad()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._init_conv_mask(name, module)
                if not self.use_greedy:
                    prune_indices_ = self._get_prune_indices(name, module, prune_rates[i], mode=mode)
                if dim == 1:
                    # self._prune_by_indices(module, prune_indices, dim=dim)
                    # self._set_conv_mask(name, prune_indices, dim=dim)
                    dim = 0
                if self.use_greedy:
                    prune_indices = self._get_prune_indices(name, module, prune_rates[i], mode=mode)
                else:
                    prune_indices = prune_indices_
                self._prune_by_indices(module, prune_indices, dim=dim)
                self._set_conv_mask(name, prune_indices, dim=dim)
                dim = 1
                i += 1
            elif isinstance(module, torch.nn.BatchNorm2d):
                if 'filter' in mode and dim == 1:
                    self._prune_by_indices(module, prune_indices, dim=0)
        if self._get_use_grad():
            self.optimizer.zero_grad()

    def _get_actual_prune_rates(self, prune_rates, verbose=False):
        """
        # Suppose the model prunes some filters (filters, :, :, :).
        # ----------------------------------------------------------
        # If the filter of the previous layer is prune, the channel corresponding to this layer must also be
        # prune
        # ----------------------------------------------------------
        # Suppose Conv Shape: (n_f, n_c, h, w)
        #
        # Then:
        #   prune_rate = ((n_f * act_prune_rate) * (n_c - n_prune_f) * n_f * kw) / (n_f * n_c * h * w)
        #
        $ Then:
        #   act_prune_rate = prune_rate * (n_c / (n_c - n_prune_f))
        #
        # "n_prune_f" is the number of prune filters last layer
        # ----------------------------------------------------------
        """
        i = 0
        n_prune_f = None
        act_prune_rates = list()
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                n_f, n_c = module.weight.shape[0:2]
                if i == 0:
                    act_prune_rate = prune_rates[i]
                else:
                    act_prune_rate = prune_rates[i] * (n_c / (n_c - n_prune_f))
                act_prune_rate = np.clip(act_prune_rate, 0.0, 1.0)
                n_prune_f = round(n_f * (1.0 - act_prune_rate))
                bias = abs(n_prune_f / n_f - act_prune_rate)
                act_prune_rates.append(act_prune_rate)
                i += 1
                if verbose:
                    print(f'{name:6} | original filter nums: {n_f:4} | prune filter nums: '
                          f'{n_prune_f:4} | actual filter prune rate : {act_prune_rate * 100.:.2f}% | '
                          f' bias: {bias * 100.:.2f}%')
        return act_prune_rates

    @staticmethod
    def get_left_dict(model):
        d = dict()
        for name, param in model.state_dict().items():
            if len(param.shape) == 4:  # Only consider conv layers
                w = param.data.cpu().numpy()
                trans_w = np.transpose(w, (1, 0, 2, 3))
                sum_f = np.sum(w.reshape(w.shape[0], -1), axis=1)
                sum_c = np.sum(trans_w.reshape(trans_w.shape[0], -1), axis=1)
                left_f_ind = np.where(sum_f != 0)[0]  # Indices of the left filters
                left_c_ind = np.where(sum_c != 0)[0]  # Indices of the left channels
                left_w = w[left_f_ind[:, None], left_c_ind]
                d[name] = (np.float32(left_w), np.int32(left_f_ind), np.int32(left_c_ind))
        return d

    def get_conv_mask(self):
        return self.conv_mask

    def prune(self, mode, prune_rates):
        prune_rates = self._check_prune_rates(prune_rates)
        is_prune_f = 'filter' in mode
        if is_prune_f:
            if self.use_actPR:
                prune_rates = self._get_actual_prune_rates(prune_rates)
            self._set_use_grad(val='-g-' in mode)
            self._prune_filters_and_channels(prune_rates, mode=mode)
        else:
            self._prune_by_percentile(prune_rates)







