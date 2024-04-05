from lib.self_attention import SimpleAttention

from modules.edge_train import *

from utils.metamodel import MetaModel
from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step

class GATCF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super(GATCF, self).__init__(user_num, serv_num, args)
        self.args = args
        self.serv_num = serv_num
        self.k = 5
        # 全局图
        userg = pickle.load(open('./datasets/data/userg.pk', 'rb'))
        servg = pickle.load(open('./datasets/data/servg.pk', 'rb'))
        #引入边缘训练矩阵
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension

        with open(f'../datasets/data/partition/sub/updated_user_embeddings.pk', 'wb') as file:
            self.agg_user_embeds = pickle.load(file)
        with open(f'../datasets/data/partition/sub/updated_item_embeddings.pk', 'wb') as file:
            self.agg_item_embeds = pickle.load(file)
        self.final_user_embeds, self.final_serv_embeds = self.get_final_embedding( self.agg_user_embeds, self.agg_item_embeds)
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

def get_final_embedding(self, agg_user_embeds, agg_item_embeds):
    combined_embeddings_array = np.array(agg_user_embeds)
    # 获取按照用户ID排序的索引
    sorted_indices = np.argsort(combined_embeddings_array[:, 0])
    print(sorted_indices[:])
    # 使用这些索引来对整个矩阵排序
    final_user_embeds = combined_embeddings_array[sorted_indices]
    # 创建模型实例
    attention_layer = SimpleAttention(self.dim)
    # 应用注意力层
    attention_result = attention_layer(agg_item_embeds)
    # 假设我们想将结果拆分回原来的五个部分并求平均
    split_attention_results = attention_result.view(self.k, self.serv_num, self.dim)
    final_serv_embeds = torch.mean(split_attention_results, dim=0)
    return final_user_embeds, final_serv_embeds



