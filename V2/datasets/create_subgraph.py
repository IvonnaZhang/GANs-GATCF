

import dgl as d
import torch
import pandas as pd
from GATCF import FeatureLookup


def subgrapph_attention(self,args):
    self.arg = args
    self.dim = args.dimension
    subuser_embeds_list = []
    for i in range(1,6):
        user_graph_path = f'userlist_group_{i}.csv'
        #print(self.user_graph_path )
        subusergraph= self.create_graph_user(user_graph_path)
        subuser_embeds = torch.nn.Embedding(subusergraph.number_of_nodes(), self.dim)
        # 为用户和服务创建嵌入层self.user_embeds和self.serv_embeds，其维度由args.dimension指定
        # 对嵌入层的权重进行Kaiming正态初始化
        # 这是一种常用的权重初始化方法，有助于防止深层网络中的梯度消失或爆炸问题
        torch.nn.init.kaiming_normal_(subuser_embeds.weight)
        # 这些层的参数包括图、维度、隐藏层大小、dropout率、注意力机制的alpha参数、头数（用于多头注意力），以及其他参数
        subusergraph_embeddings = SpGAT(self.subusergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        # 存储每个子图的处理结果
        subuser_embeds_list.append(subusergraph_embeddings)
    concatenated_embeddings = torch.cat(subuser_embeds_list, dim=0)  # 确保拼接的维度是0
    return concatenated_embeddings
def create_graph_user(user_graph_path):
    userg = d.graph([])
    ufile = pd.read_csv(user_graph_path)
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines
    user_lookup = FeatureLookup()
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
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)
        # 为每个图添加自环
        # 图神经网络中常见的做法，允许节点在信息传递过程中考虑自身的特征
    userg = d.add_self_loop(userg)
    # 将图转换为无向图
    # 确保信息可以在图中自由流动，而不受边方向的限制
    userg = d.to_bidirected(userg)
    return  userg
