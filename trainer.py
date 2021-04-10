import os
import json
import numpy as np
from tqdm import tqdm
from abc import abstractmethod
from util import get_average_meters, save_model
import torch


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

    def _get_save_model_path(self, i):
        return os.path.join(self.save_dir, f'model_epochs_{i}.pt')

    def _train_epoch(self, global_step):
        self.model.train()  # train mode
        e_loss, e_top1, e_top5 = get_average_meters(n=3)
        iter_bar = tqdm(self.train_data_iter)
        self.adjust_learning_rate()
        for i, batch in enumerate(iter_bar):
            batch = [t.to(self.device) for t in batch]

            self.optimizer.zero_grad()
            b_loss, b_top1, b_top5 = self.get_loss(batch, global_step)
            b_loss.backward()
            self.optimizer.step()

            global_step += 1
            e_loss.update(b_loss.item(), len(batch))
            e_top1.update(b_top1.item(), len(batch))
            e_top5.update(b_top5.item(), len(batch))
            iter_bar.set_description(
                f'Iter (loss={e_loss.mean:5.3f} | top1={e_top1.mean:5.3} | top5={e_top5.mean:5.3})'
            )

    def _eval_epoch(self):
        self.model.eval()  # evaluation mode
        iter_bar = tqdm(self.eval_data_iter, desc='Iter')
        e_result_vals = None
        for i, batch in enumerate(iter_bar, start=1):
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # evaluation without gradient calculation
                b_result_dict = self.evaluate(batch)  # accuracy to print
                b_result_vals = np.array(list(b_result_dict.values()))
            if e_result_vals is None:
                e_result_vals = [0] * len(b_result_vals)
            e_result_vals += b_result_vals
            iter_bar.set_description('Iter')
        e_result_dict = dict(zip(b_result_dict.keys(), e_result_vals/len(iter_bar)))
        print(e_result_dict)
        return e_result_dict

    def train(self):
        """ Train Loop """
        self.model.train()  # train mode
        self.model = self.model.to(self.device)
        global_step = 0  # global iteration steps regardless of epochs
        best_top1 = 0.
        for epoch in range(self.args.n_epochs):
            self.cur_epoch = epoch
            self._train_epoch(global_step)
            eval_result = self._eval_epoch()
            if best_top1 < eval_result['top1']:
                best_top1 = eval_result['top1']
                save_model(self.model, self._get_save_model_path(epoch))
        save_model(self.model, self._get_save_model_path(self.args.n_epochs))

    def eval(self):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.model = self.model.to(self.device)
        self._eval_epoch()

    def adjust_learning_rate(self):
        if self.cur_epoch in self.args.schedule:
            self.cur_lr *= self.args.lr_drop
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cur_lr

    @abstractmethod
    def get_loss(self, batch, global_step):
        return NotImplementedError

    @abstractmethod
    def evaluate(self, batch):
        return NotImplementedError
