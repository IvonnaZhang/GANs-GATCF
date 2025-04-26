# -*- coding: utf-8 -*-
# Author : yuxiang Zeng


import time
import torch
from utils.metamodel import MetaModel


class LightDnn(torch.nn.Module):
    def __init__(self, dim):
        super(LightDnn, self).__init__()
        self.dim = dim
        self.transfer = torch.nn.Linear(self.dim, self.dim)

    def forward(self, a, b):
        outputs = a * b
        outputs = self.transfer(outputs)
        outputs = outputs.sum(dim=-1).sigmoid()
        return outputs


class CF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super().__init__(user_num, serv_num, args)
        self.user_num = user_num
        self.serv_num = serv_num
        self.dim = args.dimension

        self.user_embeds = torch.nn.Embedding(user_num, self.dim)
        self.serv_embeds = torch.nn.Embedding(serv_num, self.dim)
        self.interaction = LightDnn(self.dim)

    def forward(self, inputs, train=True):
        userIdx, servIdx = inputs
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.serv_embeds(servIdx)
        estimated = self.interaction(user_embeds, serv_embeds)
        return estimated

    def prepare_test_model(self):
        pass

