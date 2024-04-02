# -*- coding: utf-8 -*-
# Author : yuxiang Zeng
import time
import torch
import pickle
from tqdm import tqdm

from lib.load_dataset import get_exper
from lib.parsers import get_parser
from modules.SpGAT import SpGAT
from modules.edge_train import *
from utils.datamodule import DataModule
from utils.metamodel import MetaModel
from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step


class GATCF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super(GATCF, self).__init__(user_num, serv_num, args)
        self.args = args
        # Initialize
        exper = get_exper(args)
        dataModule = DataModule(exper, args)
        userg = pickle.load(open('./datasets/data/userg.pk', 'rb'))
        servg = pickle.load(open('./datasets/data/servg.pk', 'rb'))
        #引入边缘训练
        edge = EdgeModel(args)
        self.edge_model = edge.edge_train_one_epoch(dataModule)
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.final_user_embeds, self.final_serv_embeds = self.get_final_embedding()

        torch.nn.init.kaiming_normal_(self.final_user_embeds.weight)
        torch.nn.init.kaiming_normal_(self.final_serv_embeds.weight)

        # 这些层的参数包括图、维度、隐藏层大小、dropout率、注意力机制的alpha参数、头数（用于多头注意力），以及其他参数
        self.user_attention = SpGAT(self.usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.item_attention = SpGAT(self.servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        # 定义一个多层感知机（MLP），用于在注意力机制之后进一步处理数据
        # 这个序列包含线性层、层归一化（LayerNorm）、ReLU激活函数，最终输出维度为1
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        # 分别为嵌入层、注意力层、和MLP层设置优化器
        self.cache = {}
        self.optimizer_embeds = get_optimizer(self.get_embeds_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.optimizer_tf = get_optimizer(self.get_attention_parameters(), lr=4e-3, decay=args.decay, args=args)
        self.optimizer_mlp = get_optimizer(self.get_mlp_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.scheduler_tf = torch.optim.lr_scheduler.StepLR(self.optimizer_tf, step_size=args.lr_step, gamma=0.50)

    # 向前传播
    def forward(self, inputs, train):
        # 根据rtMatrix.txt得到inputs
        userIdx, itemIdx = inputs
        if train:
            # 使用 EdgeModel 获取更新的用户和项嵌入
            user_embeds, serv_embeds = self.final_user_embeds, self.final_serv_embeds
            user_embeds = self.user_attention(user_embeds)[userIdx]
            serv_embeds = self.item_attention(serv_embeds)[itemIdx]

            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        else:
            user_embeds = self.cache['user'][userIdx]
            serv_embeds = self.cache['serv'][itemIdx]

            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        return  estimated

    def prepare_test_model(self):
        # 生成并处理用户和服务的嵌入向量，然后缓存
        Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
        user_embeds = self.user_embeds(Index)
        Index = torch.arange(self.servgraph.number_of_nodes()).cuda()
        serv_embeds = self.item_embeds(Index)
        # 应用注意力层并选择特定索引的嵌入向量进行缓存
        user_embeds = self.user_attention(user_embeds)[torch.arange(339).cuda()]
        serv_embeds = self.item_attention(serv_embeds)[torch.arange(5825).cuda()]
        self.cache['user'] = user_embeds
        self.cache['serv'] = serv_embeds

    def get_embeds_parameters(self):
        parameters = []
        # 收集用户嵌入层的参数
        for params in self.user_embeds.parameters():
            parameters += [params]
        # 收集服务嵌入层的参数
        for params in self.item_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        # 收集用户注意力层的参数
        for params in self.user_attention.parameters():
            parameters += [params]
        # 收集服务注意力层的参数
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        # 收集多层感知机（MLP）层的参数
        for params in self.layers.parameters():
            parameters += [params]
        return parameters

    def train_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            pred = self.forward(inputs, True)
            if self.args.device == 'cuda':
                inputs, value = to_cuda(inputs, value, self.args)
            # print(type(pred))
            loss = self.loss_function(pred[0].to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer_embeds, self.optimizer_tf)
            optimizer_zero_grad(self.optimizer_mlp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.30)
            optimizer_step(self.optimizer_embeds, self.optimizer_tf)
            optimizer_step(self.optimizer_mlp)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler_tf)
        return loss, t2 - t1





