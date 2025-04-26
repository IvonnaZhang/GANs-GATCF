import time
from utils.metamodel import MetaModel
from tqdm import tqdm

from modules.SpGAT import create_user_graph, SpGAT
from modules.get_embedding import *
import torch
from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step


class EdgeModel(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super(EdgeModel, self).__init__(user_num, serv_num, args)
        self.args = args
        self.serv_num = serv_num
        self.dim = args.dimension
        self.agg_user_embeds = []
        self.agg_item_embeds = []

        user_graph_path = self.args.path + f'partition/userlist_group_1.csv'
        self.sub_usergraph = create_user_graph(user_graph_path)
        self.sub_servgraph= pickle.load(open('./datasets/data/servg.pk', 'rb'))
        self.sub_user_embeds = torch.nn.Embedding(self.sub_usergraph.number_of_nodes(), self.dim)
        self.sub_item_embeds = torch.nn.Embedding(self.sub_servgraph.number_of_nodes(), self.dim)
        torch.nn.init.kaiming_normal_(self.sub_user_embeds.weight)
        torch.nn.init.kaiming_normal_(self.sub_item_embeds.weight)

        self.user_attention = SpGAT(self.sub_usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.item_attention = SpGAT(self.sub_servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.optimizer_embeds = get_optimizer(self.get_embeds_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.optimizer_tf = get_optimizer(self.get_attention_parameters(), lr=4e-3, decay=args.decay, args=args)
        self.scheduler_tf = torch.optim.lr_scheduler.StepLR(self.optimizer_tf, step_size=args.lr_step, gamma=0.50)

    def forward(self, inputs, train=True):
        userIdx, itemIdx = inputs

        # 生成一个包含所有用户/服务节点索引的Tensor
        Index = torch.arange(self.sub_usergraph.number_of_nodes())
        # 将其传递给用户/服务嵌入层self.user/serv_embeds以获取所有用户/服务的嵌入向量
        user_embeds = self.sub_user_embeds(Index)
        # Index = torch.arange(self.sub_servgraph.number_of_nodes()) # 使用方法将索引Tensor移动到GPU上
        item_embeds = self.sub_item_embeds(Index)
        # print(user_embeds.size())
        # print(item_embeds.size())
        # user_embeds = self.user_attention(user_embeds)[userIdx]
        # item_embeds = self.item_attention(item_embeds)[itemIdx]
        # print(user_embeds.shape, item_embeds.shape)
        estimated = self.layers(torch.cat((user_embeds, item_embeds), dim=-1)).sigmoid().reshape(-1)
        return estimated

    def get_embeds_parameters(self):
        parameters = []
        # 收集用户嵌入层的参数
        for params in self.sub_user_embeds.parameters():
            parameters += [params]
        # 收集服务嵌入层的参数
        for params in self.sub_item_embeds.parameters():
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

    def train_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        preds = []
        reals = []
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            # # 这里可以打印看看
            # print("input")
            # print(inputs)
            # print("value")
            # print(value)
            # print(inputs[0].shape)
            pred = self.forward(inputs, True)
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

        return loss, t2 - t1

    def prepare_test_model(self):
       pass
