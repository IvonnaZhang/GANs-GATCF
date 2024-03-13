# -*- coding: utf-8 -*-
# Author : yuxiang Zeng


import time
import torch

from tqdm import *
from abc import ABC, abstractmethod

from utils.metrics import error_metrics
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, to_cuda


class MetaModel(torch.nn.Module, ABC):
    def __init__(self, user_num, serv_num, args):
        super(MetaModel, self).__init__()
        self.args = args

    @abstractmethod
    def forward(self, inputs, train = True):
        pass

    @abstractmethod
    def prepare_test_model(self):
        pass

    def setup_optimizer(self, args):
        if args.device != 'cpu':
            self.to(args.device)
            self.loss_function = get_loss_function(args).to(args.device)
        else:
            self.loss_function = get_loss_function(args)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=0.50)


    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value, self.args)
            pred = self.forward(inputs, True)
            loss = self.loss_function(pred.to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler)

        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device) if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device) if self.args.device != 'cpu' else torch.zeros((len(dataModule.valid_loader.dataset),))
        self.prepare_test_model()
        for valid_Batch in tqdm(dataModule.valid_loader, disable=not self.args.program_test):
            inputs, value = valid_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value, self.args)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        valid_error = error_metrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device) if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device) if self.args.device != 'cpu' else torch.zeros((len(dataModule.test_loader.dataset),))
        self.prepare_test_model()
        for test_Batch in tqdm(dataModule.test_loader, disable=not self.args.program_test):
            inputs, value = test_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value, self.args)
            pred = self.forward(inputs, False)
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        test_error = error_metrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error
