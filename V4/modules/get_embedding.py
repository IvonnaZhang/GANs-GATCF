### 得到embeddings
# coding : utf-8
# Author : Yuxin Zhang
import pickle
import pickle as pk
import dgl as d
import pandas as pd
import numpy as np
from node2vec import Node2Vec

# node2vec_dim, node2vec_length, node2vec_walk, node2vec_epochs, node2vec_batchsize
# 128           5                50             20               32


# Node2vec get pretrain
def get_user_embeds(args, path, x):
    #图的构建
    # 初始化图和用户查找表
    userg = d.graph([]) # 空图userg
    user_lookup = FeatureLookup() # FeatureLookup实例user_lookup

    ## 读取用户数据
    # 加载文件并转换成numpy数组
    ufile = pd.read_csv(path)
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    row = ufile.shape[0]
    # print(row)

    ## 注册用户和特征
    for i in range(row):
        user_lookup.register('User', i) # 注册每个用户
    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure) # 遍历用户的两种特征（URE和UAS）并注册
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas) # 遍历用户的两种特征（URE和UAS）并注册

    # 添加节点和边
    # 为图userg添加节点，节点数量等于user_lookup中注册的项数
    userg.add_nodes(len(user_lookup))

    # 遍历用户数据，为每个用户添加与其URE和UAS特征对应的边
    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])

        # 这里使用user_lookup.query_id查询特征对应的内部ID，并检查是否已存在边，如果不存在，则添加边
        if not userg.has_edges_between(i, ure):
            userg.add_edges(i, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(i, uas):
            userg.add_edges(i, uas)

    ## 图转换
    # 通过d.add_self_loop(userg)为每个节点添加自环，增强模型的特征学习能力
    userg = d.add_self_loop(userg)
    # 使用d.to_bidirected(userg)将图转换为双向图，以便在随机游走时能够在任意方向上移动
    userg = d.to_bidirected(userg)

    ## 转换为NetworkX图
    # 将图转换为NetworkX图对象G，以便使用Node2Vec库
    G = userg.to_networkx()

    ### Node2Vec参数设置与训练
    ## 初始化Node2Vec
    # 使用图G和从args提取的参数初始化Node2Vec模型
    node2vec = Node2Vec(
                        G,
                        dimensions=args.node2vec_dim,  # 嵌入维度
                        p=1,  # 回家参数
                        q=0.5,  # 外出参数
                        walk_length=args.node2vec_length,  # 随机游走最大长度
                        num_walks=args.node2vec_walk,  # 每个节点作为起始节点生成的随机游走个数
                        workers=1,  # 并行线程数
                        seed=args.random_state
                        )

    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2, n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    ## 训练Node2Vec
    # 通过调用node2vec.fit方法训练模型
    # 传入Skip-Gram窗口大小、训练轮次、最小出现次数、每批处理的单词数量和随机状态
    model = node2vec.fit(
                         window=args.node2vec_windows,  # Skip-Gram窗口大小
                         epochs=args.node2vec_epochs,
                         min_count = 3,  # 忽略出现次数低于此阈值的节点（词）
                         batch_words=args.node2vec_batchsize,  # 每个线程处理的数据量
                         seed = args.random_state
                        )

    ### 用户嵌入的保存
    # 从训练好的模型中提取前339个向量，这些向量代表用户的嵌入表示
    ans = model.wv.vectors[:row]
    user_embedding = np.array(ans)

    print("用户嵌入向量的形状:", user_embedding.shape)
    print("前几个用户嵌入向量的样本:", user_embedding[:5])

    # 将用户嵌入向量保存为一个pickle文件，以便后续使用
    if 'group' in path:
        pk.dump(user_embedding, open(f'datasets/data/partition/sub/subuser_embeds_{x}.pk', 'wb'))
    else:
        pk.dump(user_embedding, open(f'datasets/data/embeddings/user_embeds.pk', 'wb'))

    return user_embedding
def get_user_embedding(args,path):
    with open(path, 'rb') as file:
        embeddings = pickle.load(file)
    sub_user_embeds = embeddings[:][1:]
    return sub_user_embeds

def get_item_embedding(args):
    servg = d.graph([])

    serv_lookup = FeatureLookup()
    sfile = pd.read_csv('./datasets/data/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()

    for i in range(5825):
        serv_lookup.register('Sid', i)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)

    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)

    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    servg.add_nodes(len(serv_lookup))

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

    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)

    G = servg.to_networkx()
    # 设置node2vec参数
    node2vec = Node2Vec(G,
                        dimensions=args.node2vec_dim,  # 嵌入维度
                        p=1,  # 回家参数
                        q=0.5,  # 外出参数
                        walk_length=args.node2vec_length,  # 随机游走最大长度
                        num_walks=args.node2vec_walk,  # 每个节点作为起始节点生成的随机游走个数
                        workers=1  # 并行线程数
                        )

    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2, n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    # 训练Node2Vec，参数文档见 gensim.models.Word2Vec
    model = node2vec.fit(window=args.node2vec_windows,  # Skip-Gram窗口大小
                         epochs=args.node2vec_epochs,
                         min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                         batch_words=args.node2vec_batchsize,  # 每个线程处理的数据量
                         seed = args.random_state
                         )

    ans = model.wv.vectors[:5825]
    item_embedding = np.array(ans)

    print("项目嵌入向量的形状:", item_embedding.shape)
    print("前几个项目嵌入向量的样本:", item_embedding[:5])

    # 将项目嵌入向量保存为一个pickle文件，以便后续使用
    pk.dump(item_embedding, open(f'datasets/data/embeddings/item_embeds.pk', 'wb'))

    return item_embedding


class FeatureLookup:
    # 初始化所有的私有属性
    def __init__(self):
        self.__inner_id_counter = 0 # 初始化唯一ID计数器
        self.__inner_bag = {} # 初始化全局值到ID的映射
        self.__category = set() # 初始化类别集合
        self.__category_bags = {} # 初始化每个类别的值到ID的映射
        self.__inverse_map = {} # 初始化ID到值的反向映射

    # 注册一个新的值和其对应的类别
    def register(self, category, value):
        # 将新类别添加到类别集合中
        self.__category.add(category)

        # 如果类别不存在，则新增一个类别子树（子字典不存在，则创建一个空字典）
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则进行注册，id += 1
        if value not in self.__inner_bag:
            # 为值分配一个唯一的整数ID
            self.__inner_bag[value] = self.__inner_id_counter
            # 创建从ID到值的反向映射
            self.__inverse_map[self.__inner_id_counter] = value

            # 如果值不存在与类别子树，则创建之（值在当前类别的子字典中尚未注册，则进行注册）
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            # 递增ID计数器，为下一个值准备
            self.__inner_id_counter += 1

    # 根据提供的值返回其唯一整数ID
    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    # 根据提供的唯一整数ID返回原始值
    def query_value(self, idx):
        # 返回值
        return self.__inverse_map[idx]

    # 返回__inner_bag中元素的数量，即注册的唯一值的总数
    def __len__(self):
        return len(self.__inner_bag)