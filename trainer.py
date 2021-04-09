import os
import json
import numpy as np
from typing import NamedTuple
from tqdm import tqdm
from abc import abstractmethod
from util import get_average_meters
import torch
import torch.nn as nn


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, args, model, train_data_iter, eval_data_iter, optimizer, save_dir, device):
        self.args = args
        self.model = model
        self.train_data_iter = train_data_iter  # iterator to load data
        self.eval_data_iter = eval_data_iter  # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device  # device name

        self.cur_epoch = None
        self.cur_lr = args.lr

    def _train_epoch(self, model, global_step):
        self.model.train()  # train mode
        # loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
        e_loss, e_top1, e_top5 = get_average_meters(n=3)
        iter_bar = tqdm(self.train_data_iter)
        self.adjust_learning_rate()
        for i, batch in enumerate(iter_bar):
            batch = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            b_loss, b_top1, b_top5 = self.get_loss(model, batch, global_step)
            b_loss.backward()
            self.optimizer.step()

            global_step += 1
            e_loss.update(b_loss.item(), len(batch))
            e_top1.update(b_top1.item(), len(batch))
            e_top5.update(b_top5.item(), len(batch))
            iter_bar.set_description(
                f'Iter (loss={e_loss.mean:5.3f} | top1={e_top1.mean:5.3} | top5={e_top5.mean:5.3})'
            )

    def _eval_epoch(self, model):
        self.model.eval()  # evaluation mode
        iter_bar = tqdm(self.eval_data_iter, desc='Iter')
        epoch_result_vals = None
        for i, batch in enumerate(iter_bar, start=1):
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                batch_result_dict = self.evaluate(model, batch)  # accuracy to print
                batch_result_vals = np.array(list(batch_result_dict.values()))
            if epoch_result_vals is None:
                epoch_result_vals = [0] * len(batch_result_vals)
            epoch_result_vals += batch_result_vals
            iter_bar.set_description('Iter')
        epoch_result_dict = dict(zip(batch_result_dict.keys(), epoch_result_vals/len(iter_bar)))
        print(epoch_result_dict)
        return epoch_result_dict

    def train(self, model_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train()  # train mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel:  # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        global_step = 0  # global iteration steps regardless of epochs
        best_top1 = 0.
        for epoch in range(self.args.n_epochs):
            self.cur_epoch = epoch
            self._train_epoch(model, global_step)
            eval_dict = self._eval_epoch(model)
            if best_top1 < eval_dict['top1']:
                best_top1 = eval_dict['top1']
                self.save(epoch)
        self.save(self.args.n_epochs)

    def eval(self, model_file=None, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)
        self._eval_epoch(model)

    def load(self, model_file):
        """ Load saved model or pretrained transformer (a part of model) """
        print('Loading the model from', model_file)
        if model_file is not None:
            self.model.load_state_dict(torch.load(model_file, map_location=self.device), strict=False)

    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_epochs_' + str(i) + '.pt'))

    def adjust_learning_rate(self):
        if self.cur_epoch in self.args.schedule:
            self.cur_lr *= self.args.lr_drop
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cur_lr

    @abstractmethod
    def get_loss(self, model, batch, global_step):
        return NotImplementedError

    @abstractmethod
    def evaluate(self, model, batch):
        return NotImplementedError
