import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl as d
import pandas as pd

class SpGAT(torch.nn.Module):
    def __init__(self, graph, nfeat, nhid, dropout, alpha, nheads, args):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout # 存储dropout比率

        # 调用get_adj_norm_matrix静态方法获取图的邻接矩阵，并将其标准化和转移到CUDA上
        self.adj = self.get_adj_nrom_matrix(graph)
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
        n = g.number_of_nodes()
        in_deg = g.in_degrees().numpy()
        rows = g.edges()[1].numpy()
        cols = g.edges()[0].numpy()
        adj = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

        def normalize_adj(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))  # 求每一行的和
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-0.5}
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # D^{-0.5}
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # adj = D^{-0.5}SD^{-0.5}, S=A+I
        print(adj.dtype)
        print(adj)
        adj = torch.FloatTensor( adj.toarray())
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

    def __init__(self, in_features, out_features, dropout, alpha, concat = True):
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

def create_user_graph(user_graph_path):
    userg = d.graph([])
    user_lookup = FeatureLookup()

    ufile = pd.read_csv(user_graph_path)
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    # 进一步遍历用户数据
    for i in ulines[:, 0]:
        user_lookup.register('User', i)  # 用户ip地址

    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)  # 用户评价分数User Rating Score

    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)  # 用户活跃度分数User Activity Score

    userg.add_nodes(len(user_lookup))

    # 迭代用户和服务数据，根据特征之间的关系添加边
    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(i, ure):
            userg.add_edges(i, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(i, uas):
            userg.add_edges(i, uas)

    # 为每个图添加自环
    # 图神经网络中常见的做法，允许节点在信息传递过程中考虑自身的特征
    userg = d.add_self_loop(userg)

    # 将图转换为无向图
    # 确保信息可以在图中自由流动，而不受边方向的限制
    userg = d.to_bidirected(userg)

    return  userg


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