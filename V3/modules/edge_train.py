from abc import abstractmethod, ABC

import dgl as d
import torch as t
import pandas as pd
from torch.nn import *
from GATCF import *
from lib.load_dataset import get_exper
from modules import get_model
from modules.aggregator import cat_embedding
from modules.get_embedding import get_user_embedding, get_item_embedding
from utils.datamodule import DataModule


class EdgeModel(torch.nn.Module, ABC):
    def __init__(self, user_num, serv_num, args):
        super(EdgeModel, self).__init__()
        self.args = args

    @abstractmethod
    def forward(self, inputs, train = True):
        pass

    def edge(self, args):
        self.arg = args
        self.dim = args.dimension
        exper = get_exper(args)
        dataModule = DataModule(exper, args)
        model = get_model(dataModule, args)

        # 建子图
        user_embeds_list = []
        item_embeds_list = []

        for x in range(1,6):
            user_graph_path = f'userlist_group_{x}.csv'
            # print(self.user_graph_path)
            # subuser_graph= self.create_user_graph(user_graph_path)
            # subuser_embeds = torch.nn.Embedding(subuser_graph.number_of_nodes(), self.dim)
            subuser_embeds_x = get_user_embedding(args, user_graph_path)
            user_embeds_x, item_embeds_x = model.edge_train(dataModule)

            # 将生成的嵌入添加到列表中
            user_embeds_list.append(user_embeds_x)
            item_embeds_list.append(item_embeds_x)

        new_user_embed = self.agg(user_embeds_list)
        new_item_embed = self.att(item_embeds_list)

        return new_user_embed, new_item_embed

    def edge_train(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            if self.args.device != 'cpu':
                inputs, value = to_cuda(inputs, value, self.args)
            pred = self.forward(inputs, True)
            user_embeds, serv_embeds = self.forward(inputs, True)
            loss = self.loss_function(pred.to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler)

        return loss, t2 - t1, user_embeds, serv_embeds

class UserAggregator(Module):
    def __init__(self, datasets, dim, args):
        super().__init__()
        self.datasets = datasets
        self.args = args
        self.dim = dim

    def forward(self, user_embeds_list, userIdx):
        # [num_slices, num_users, embed_dim] -> [num_users, embed_dim]
        embeds = t.as_tensor(user_embeds_list)
        agg_embeds = cat_embedding(embeds, self.datasets, userIdx, 'user', self.args)

        return agg_embeds

    def att(self, item_embeds_list, itemIdx):
        embeds = t.as_tensor(item_embeds_list)
        agg_embeds = self.attention(embeds)  # external attention [bs, n, dim] ->
        agg_embeds = agg_embeds.permute(1, 0, 2).reshape(-1, self.dim * self.args.slices)
        agg_embeds = self.fc(agg_embeds)

        return agg_embeds


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

    return user_lookup, userg