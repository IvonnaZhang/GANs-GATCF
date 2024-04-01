import pickle
from abc import abstractmethod, ABC
from datetime import time

from tqdm import tqdm

from modules.SpGAT import create_user_graph, SpGAT
from modules.get_embedding import *
import numpy as np
from torch.nn import *
import torch

from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step


class EdgeModel(torch.nn.Module, ABC):
    def __init__(self, args):
        super(EdgeModel, self).__init__()
        self.args = args

        self.agg_user_embeds = []
        self.agg_item_embeds = []

        for x in range(1, 6):
            self.sub_round = x
            user_graph_path = self.args.path + f'partition/userlist_group_{x}.csv'
            self.sub_usergraph = create_user_graph(user_graph_path)
            servg = pickle.load(open('./datasets/data/servg.pk', 'rb'))
            self.sub_servgraph = servg
            user_embedding_path = self.args.path + f'partition/sub/subuser_embeds_{x}.pk'
            pretrained_user_matrix = get_subuser_embedding(args, user_embedding_path)
            num_embeddings, embedding_dim = pretrained_user_matrix.shape
            self.pretrained_user_embeds = torch.nn.Embedding(num_embeddings, embedding_dim)
            self.pretrained_user_embeds.weight = torch.nn.Parameter(pretrained_user_matrix )
            # 如果你不希望在训练中更新这些嵌入，可以将它们设置为不需要梯度
            self.pretrained_user_embeds.weight.requires_grad = True # 如果需要微调

            pretrained_item_embeds = get_item_embedding(args)

            self.pretrained_item_embeds = torch.nn.Embedding(5825, embedding_dim)
            self.pretrained_item_embeds.weight.requires_grad = True

            self.user_attention = SpGAT(self.sub_usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
            self.item_attention = SpGAT(self.sub_servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)

            self.optimizer_embeds = get_optimizer(self.get_embeds_parameters(), lr=1e-2, decay=args.decay, args=args)
            self.optimizer_tf = get_optimizer(self.get_attention_parameters(), lr=4e-3, decay=args.decay, args=args)
            self.scheduler_tf = torch.optim.lr_scheduler.StepLR(self.optimizer_tf, step_size=args.lr_step, gamma=0.50)

    def forward(self, inputs, train=True):
        userIdx, itemIdx = inputs

        # 生成一个包含所有用户/服务节点索引的Tensor
        Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
        # 将其传递给用户/服务嵌入层self.user/serv_embeds以获取所有用户/服务的嵌入向量
        user_embeds = self.pretrained_user_embeds(Index)
        Index = torch.arange(self.servgraph.number_of_nodes()).cuda() # 使用.cuda()方法将索引Tensor移动到GPU上
        item_embeds = self.pretrained_item_embeds(Index)

        user_embeds = self.user_attention(user_embeds)[userIdx]
        item_embeds = self.item_attention(item_embeds)[itemIdx]

        estimated = self.layers(torch.cat((user_embeds, item_embeds), dim=-1)).sigmoid().reshape(-1)

        # agg_user_embeds, agg_item_embeds = self.agg_user_embeds, self.agg_item_embeds

        return estimated, user_embeds, item_embeds

    def get_embeds_parameters(self):
        parameters = []
        # 收集用户嵌入层的参数
        for params in self.pretrained_user_embeds.parameters():
            parameters += [params]
        # 收集服务嵌入层的参数
        for params in self.pretrained_item_embeds.parameters():
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

    def get_attention_parameters(self):
        parameters = []
        # 收集用户注意力层的参数
        for params in self.user_attention.parameters():
            parameters += [params]
        # 收集服务注意力层的参数
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def edge_train_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            #这里可以打印看看
           # print(inputs.size(),value.size())
            pred, edge_user_embeds, edge_serv_embeds = self.forward(inputs, True)
            if self.args.device == 'cuda':
                inputs, value = to_cuda(inputs, value, self.args)
            # print(type(pred))
            loss = self.loss_function(pred[0].to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer_embeds, self.optimizer_tf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.30)
            optimizer_step(self.optimizer_embeds, self.optimizer_tf)

        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler_tf)

        pk.dump(edge_user_embeds, open(f'datasets/data/partition/final/label_user_embeds_{self.sub_round}.pk', 'wb'))
        pk.dump(edge_serv_embeds, open(f'datasets/data/partition/final/label_serv_embeds_{self.sub_round}.pk', 'wb'))

        return loss, t2 - t1

    def get_final_embedding(agg_user_embeds,agg_item_embeds):
        # 首先将其转换为NumPy数组
        combined_embeddings_array = np.array(agg_user_embeds)
        # 获取按照用户ID排序的索引
        sorted_indices = np.argsort(combined_embeddings_array[:, 0])
        # 使用这些索引来对整个矩阵排序
        final_user_embeds = combined_embeddings_array[sorted_indices]
        agg_item_embeds_tensor = torch.stack(agg_item_embeds)  # 大小为 [N, 3525, 128]
        # 假设使用8个头，那么每个头的特征维度是 128 / 8 = 16
        multihead_attn = MultiheadAttention(embed_dim=128, num_heads=8)

        # 这里我们需要转置 agg_item_embeds_tensor 以符合 MultiheadAttention 的输入需求
        # 输入的形状应该是 [seq_len, batch_size, embed_dim]
        # 假设我们把3525视为序列长度，N视为批量大小
        query = key = value = agg_item_embeds_tensor.transpose(0, 1)  # 转置后大小为 [3525, N, 128]

        # 注意力层的输出大小也会是 [seq_len, batch_size, embed_dim]，即 [3525, N, 128]
        attn_output, _ = multihead_attn(query, key, value)

        # 这里我们简单地取平均
        final_serv_embeds = attn_output.mean(dim=1)

        return final_user_embeds, final_serv_embeds
