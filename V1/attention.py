# -*- coding: utf-8 -*-
# Author : yuxiang Zeng
import time

import dgl as d
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm

from utils.metamodel import MetaModel
from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step


class attention(MetaModel):
    def __init__(self, user_num, serv_num, args):
        # 调用基类 MetaModel 的构造函数
        super(attention, self).__init__(user_num, serv_num, args)
        self.args = args # 将传入的参数保存为类的属性，以便后续使用
        # 尝试从文件中加载预先保存的用户图userg和服务图servg
        try:
            userg = pickle.load(open('./modules/models/baselines/userg.pk', 'rb'))
            servg = pickle.load(open('./modules/models/baselines/servg.pk', 'rb'))
        # 如果失败（文件不存在），则调用create_graph()函数创建新的图，并将它们保存到文件中
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./modules/models/baselines/userg.pk', 'wb'))
            pickle.dump(servg, open('./modules/models/baselines/servg.pk', 'wb'))
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        # 为用户和服务创建嵌入层self.user_embeds和self.serv_embeds，其维度由args.dimension指定
        # 嵌入层的大小基于各自图中的节点数量
        self.user_embeds = torch.nn.Embedding(self.usergraph.number_of_nodes(), self.dim)
        self.serv_embeds = torch.nn.Embedding(self.servgraph.number_of_nodes(), self.dim)
        # 对嵌入层的权重进行Kaiming正态初始化
        # 这是一种常用的权重初始化方法，有助于防止深层网络中的梯度消失或爆炸问题
        torch.nn.init.kaiming_normal_(self.user_embeds.weight)
        torch.nn.init.kaiming_normal_(self.serv_embeds.weight)

        # 使用自定义的图注意力层SpGAT为用户和服务图创建注意力机制
        # 这些层的参数包括图、维度、隐藏层大小、dropout率、注意力机制的alpha参数、头数（用于多头注意力），以及其他参数
        self.user_attention = SpGAT(self.usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.serv_attention = SpGAT(self.servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
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
        # get_optimizer函数根据层的参数、学习率、衰减率以及其他参数创建优化器
        # 这表明模型训练时对这三部分可能采取不同的优化策略
        self.optimizer_embeds = get_optimizer(self.get_embeds_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.optimizer_tf = get_optimizer(self.get_attention_parameters(), lr=4e-3, decay=args.decay, args=args)
        self.optimizer_mlp = get_optimizer(self.get_mlp_parameters(), lr=1e-2, decay=args.decay, args=args)
        # 为注意力层的优化器设置学习率调度器，定期调整学习率以改善训练过程和结果
        self.scheduler_tf = torch.optim.lr_scheduler.StepLR(self.optimizer_tf, step_size=args.lr_step, gamma=0.50)

    # 向前传播
    def forward(self, inputs, train):
        # 根据rtMatrix.txt得到inputs
        userIdx, itemIdx = inputs
        if train:
            # 生成一个包含所有用户/服务节点索引的Tensor
            Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
            # 将其传递给用户/服务嵌入层self.user/serv_embeds以获取所有用户/服务的嵌入向量
            user_embeds = self.user_embeds(Index)
            Index = torch.arange(self.servgraph.number_of_nodes()).cuda()
            serv_embeds = self.item_embeds(Index)

            # 分别对用户和服务的嵌入向量应用图注意力机制，以学习节点间的复杂依赖关系
            # 注意力机制的输出结果通过索引userIdx和servIdx筛选出对应于输入索引的嵌入向量，以便进行后续的计算
            user_embeds = self.user_attention(user_embeds)[userIdx]
            serv_embeds = self.item_attention(serv_embeds)[itemIdx]

            # torch.cat((user_embeds, serv_embeds), dim=-1)的意思：
            # 将用户和服务的嵌入向量沿最后一个维度（dim = -1）拼接，以便同时考虑用户和服务的特征
            # self.layers(...).sigmoid().reshape(-1)通过多层感知机（MLP）self.layers的意思：
            # 处理拼接后的嵌入向量，使用sigmoid函数将输出激活为预测值，最后通过.reshape(-1)调整形状以确保输出是一维的，每个元素对应一个预测结果
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        else:
            user_embeds = self.cache['user'][userIdx]
            serv_embeds = self.cache['serv'][itemIdx]
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)

        # 模型预测的用户对服务的评分或偏好
        # 这个预测值经过sigmoid激活，因此其范围在0到1之间
        return estimated

    def prepare_test_model(self):
        # 生成并处理用户和服务的嵌入向量，然后缓存
        Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
        user_embeds = self.user_embeds(Index)
        Index = torch.arange(self.servgraph.number_of_nodes()).cuda()
        serv_embeds = self.serv_embeds(Index)
        # 应用注意力层并选择特定索引的嵌入向量进行缓存
        user_embeds = self.user_attention(user_embeds)[torch.arange(339).cuda()]
        serv_embeds = self.serv_attention(serv_embeds)[torch.arange(5825).cuda()]
        self.cache['user'] = user_embeds
        self.cache['serv'] = serv_embeds

    def get_embeds_parameters(self):
        parameters = []
        # 收集用户嵌入层的参数
        for params in self.user_embeds.parameters():
            parameters += [params]
        # 收集服务嵌入层的参数
        for params in self.serv_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        # 收集用户注意力层的参数
        for params in self.user_attention.parameters():
            parameters += [params]
        # 收集服务注意力层的参数
        for params in self.serv_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        # 收集多层感知机（MLP）层的参数
        for params in self.layers.parameters():
            parameters += [params]
        return parameters

    def train_one_epoch(self, dataModule):
        self.train() # 启用模型的训练模式
        torch.set_grad_enabled(True) # 启用梯度计算
        t1 = time.time() # 记录训练开始时间
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            if self.args.device == 'cuda':
                inputs, value = to_cuda(inputs, value, self.args)
            pred = self.forward(inputs, True) # 前向传播计算预测值
            # 计算损失函数
            loss = self.loss_function(pred.to(torch.float32), value.to(torch.float32))
            # 重置所有优化器的梯度
            optimizer_zero_grad(self.optimizer_embeds, self.optimizer_tf)
            optimizer_zero_grad(self.optimizer_mlp)
            loss.backward() # 反向传播计算梯度
            # 使用梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.30)
            # 更新优化器的权重
            optimizer_step(self.optimizer_embeds, self.optimizer_tf)
            optimizer_step(self.optimizer_mlp)
        t2 = time.time() # 记录训练结束时间
        self.eval() # 切换到模型的评估模式
        torch.set_grad_enabled(False) # 禁用梯度计算
        # 更新学习率调度器
        lr_scheduler_step(self.scheduler_tf)
        return loss, t2 - t1 # 返回损失和训练耗时


class SpGAT(torch.nn.Module):
    def __init__(self, graph, nfeat, nhid, dropout, alpha, nheads, args):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout # 存储dropout比率

        # 调用get_adj_norm_matrix静态方法获取图的邻接矩阵，并将其标准化和转移到CUDA上
        self.adj = self.get_adj_nrom_matrix(graph).cuda()
        self.numbers = len(self.adj)

        # 初始化一个模块列表，用于存储多个头的注意力层实例
        self.attentions = torch.nn.ModuleList()
        self.nheads = nheads

        # 为每个注意力头创建一个SpGraphAttentionLayer实例并添加到self.attentions列表中
        # 每个注意力层负责学习输入特征到隐藏层特征的映射
        for i in range(self.nheads):
            temp = SpGraphAttentionLayer(nfeat, nhid, dropout=args.dropout, alpha=alpha, concat=True)
            self.attentions += [temp]

        # 定义一个dropout层，用于在前向传播中应用dropout
        self.dropout_layer = torch.nn.Dropout(p=self.dropout, inplace=False)
        # 定义输出层的图注意力层，它将多头注意力的结果（维度为nhid * nheads）映射回输入特征空间的维度（nfeat）
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, embeds):
        x = self.dropout_layer(embeds)
        # 对于self.attentions中的每个注意力头，使用输入嵌入和邻接矩阵计算注意力结果，然后将所有头的结果沿特征维度拼接
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, self.adj))
        return x

    @staticmethod
    def get_adj_nrom_matrix(graph):
        g = graph
        # 转换为邻接矩阵
        n = g.number_of_nodes()  # 获取图中节点的数量
        in_deg = g.in_degrees().numpy()  # 计算图中每个节点的入度
        rows = g.edges()[1].numpy()  # 获取图中所有边的目标节点索引
        cols = g.edges()[0].numpy()  # 获取图中所有边的源节点索引
        # 使用rows和cols构建CSR格式的稀疏邻接矩阵，矩阵大小为n*n，矩阵中的值全部为1
        adj = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

        # 通过normalize_adj函数对邻接矩阵进行行归一化处理，以便每行的元素之和为1
        def normalize_adj(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))  # 求每一行的和
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-0.5}
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # D^{-0.5}
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        # 添加自连接
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 确保邻接矩阵是对称的，即无向图的表示
        # 归一化邻接矩阵
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # adj = D^{-0.5}SD^{-0.5}, S=A+I
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj


# 简单的图注意力网络层
class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout # dropout比率
        self.in_features = in_features # 输入特征的维度
        self.out_features = out_features # 输出特征的维度
        self.alpha = alpha # LeakyReLU的负斜率
        self.concat = concat # 一个布尔值，指示是否在多头注意力中连接输出

        # self.W是一个权重矩阵，将输入特征转换到输出特征空间
        # 它通过torch.nn.Parameter定义，允许自动梯度计算和优化
        # torch.nn.init.xavier_uniform_用于初始化self.W和注意力机制的参数self.a，以帮助模型在训练初期保持稳定
        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # 计算节点特征通过权重矩阵W转换后的结果，即应用线性变换
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 计算注意力系数的原始分数e
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        # 根据邻接矩阵adj应用掩码，以便只在实际存在边的节点间计算注意力分数
        # 不存在边的位置被设置为一个非常小的值（接近于负无穷），确保这些位置在后续的softmax中贡献极小
        attention = torch.where(adj > 0, e, zero_vec)
        # 应用softmax归一化计算最终的注意力权重
        attention = F.softmax(attention, dim=1)
        # 并通过dropout进行正则化
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 计算加权节点特征的线性组合，即注意力加权的节点表示
        h_prime = torch.matmul(attention, Wh)

        # 根据concat参数的值，选择是通过ELU激活函数返回结果，还是直接返回结果
        # 在多头注意力设置中，concat=True意味着将多个头的输出拼接在一起，而在最后一层通常设置concat=False以获得最终的节点表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    # 计算每对节点间原始的注意力分数e
    # 通过将变换后的节点特征Wh与注意力参数a相乘并应用LeakyReLU激活
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # Wh1和Wh2分别与a的前半部分和后半部分相乘
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # 然后将Wh1与Wh2的转置相加，计算所有节点对的注意力分数
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    # 提供类的字符串表示，方便调试和可视化
    # 它显示了图注意力层从输入特征到输出特征的映射维度
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    # 该方法首先断言indices不需要梯度（这是因为在大多数情况下，稀疏矩阵的结构是固定的，不参与训练）
    # 然后，使用indices和values创建一个torch.sparse_coo_tensor表示的稀疏矩阵a（这个稀疏矩阵与密集矩阵b进行乘法操作）
    # ctx.save_for_backward用于保存在反向传播中需要的张量a和b，同时保存稀疏矩阵的维度N用于反向传播的计算
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    # 计算关于稀疏矩阵values和密集矩阵b的梯度
    # 首先，它检查哪些输入需要梯度：1. 如果values需要梯度，方法计算grad_a_dense，这是稀疏矩阵a与梯度grad_output相乘的结果
    #                            然后，它使用稀疏矩阵的索引从grad_a_dense中提取对应的梯度并赋给grad_values
    #                         2. 如果b需要梯度，计算a的转置与grad_output的乘积作为grad_b
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


# 简单的封装类，使得SpecialSpmmFunction可以像标准的torch.nn模块一样在神经网络中使用
class SpecialSpmm(torch.nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


# 稀疏版本的图注意力网络层
class SpGraphAttentionLayer(torch.nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 一个线性层，用于将输入特征从in_features映射到out_features
        self.layer = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.kaiming_normal_(self.layer.weight)

        # 注意力机制的权重参数，初始化为全零，维度为1 x (2 * out_features)，之后使用Kaiming初始化方法
        self.a = torch.nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        torch.nn.init.kaiming_normal_(self.a.data)

        # 正则化和非线性激活
        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        # 一个特殊的稀疏矩阵乘法（SpMM）模块，用于高效地处理稀疏数据
        self.special_spmm = SpecialSpmm()

        # 层归一化（LayerNorm），用于稳定训练过程
        self.norm = torch.nn.LayerNorm(out_features)

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0] # 输入特征矩阵的行数，即图中的节点数
        edge = adj.nonzero().t() # 从adj中提取的非零元素索引，表示图中的边

        # 对输入特征进行线性变换
        h = self.layer(input)
        # h: N x out
        assert not torch.isnan(input).any()
        assert not torch.isnan(self.layer.weight).any()
        assert not torch.isnan(h).any()

        # 使用edge_h拼接每条边的源节点和目标节点的特征
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2 * D x E

        # 然后通过参数self.a计算每条边的注意力得分edge_e，并应用LeakyReLU和指数函数进行非线性转换
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        # 对注意力得分进行dropout处理
        edge_e = self.dropout(edge_e)
        # edge_e: E

        # 计算注意力加权的节点特征
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        # 对h_prime应用层归一化
        h_prime = self.norm(h_prime)

        # h_prime = h_prime.div(e_rowsum)
        # # h_prime: N x out
        # assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# GraphMF
def create_graph():
    userg = d.graph([])
    servg = d.graph([])
    # 用于注册和查询特征的ID
    # 为用户和服务创建各自的查找表实例
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    # 读取和处理数据
    ufile = pd.read_csv('./datasets/原始数据/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    # 将DataFrame对象转换为NumPy数组（ulines和slines）
    slines = sfile.to_numpy()
    slines = slines

    # 通过循环注册已知数量的用户（339个）和服务（5825个）到各自的查找表中
    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)

    # 进一步遍历用户和服务数据
    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure) # 用户评价分数User Rating Score
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas) # 用户活跃度分数User Activity Score

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre) # 代表服务响应时间Service Response time
    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr) # 代表服务价格Service Price
    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas) # 代表服务活跃度分数Service Activity Score

    # 根据查找表中的特征数量，为用户图和服务图添加相应数量的节点
    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    # 迭代用户和服务数据，根据特征之间的关系添加边
    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    # 为每个图添加自环
    # 图神经网络中常见的做法，允许节点在信息传递过程中考虑自身的特征
    userg = d.add_self_loop(userg)
    # 将图转换为无向图
    # 确保信息可以在图中自由流动，而不受边方向的限制
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:

    def __init__(self):
        self.__inner_id_counter = 0 # 用于生成唯一ID的计数器
        self.__inner_bag = {} # 将值映射到其唯一ID
        self.__category = set() # 存储所有注册的类别
        self.__category_bags = {} # 根据类别对值和其ID进行分组存储
        self.__inverse_map = {} # 用于通过ID反向查找值

    def register(self, category, value):
        # 添加进入类别
        self.__category.add(category)
        # 如果类别不存在若无则，则新增一个类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则创建之，id += 1
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            # 如果值不存在与类别子树，则创建之
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    def query_value(self, id):
        # 返回值
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)