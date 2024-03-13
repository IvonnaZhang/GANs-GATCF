# coding : utf-8
# Author : yuxiang Zeng
import random
import numpy as np
import pickle as pk
from tqdm import *

from lib.utils import *


### RecEarser
# 定义计算欧氏距离平方的函数
def E_score1(a, b):
    return np.sum(np.power(a - b, 2))

def E_score2(a, b):
    # 确保a和b是NumPy数组
    a = np.array(a)
    b = np.array(b)
    # 计算差的平方和
    return np.sum(np.power(a - b, 2))


## 根据交互矩阵和参数进行平衡分区
# tensor：用户-项目交互矩阵，一个二维数组，其中tensor[i][j]表示用户i和项目j之间的交互强度
# args：一个包含各种参数的对象，例如分区数量(slices)、迭代次数(part_iter)等
def interaction_based_balanced_parition(tensor, args):

    try:
        # 尝试加载预训练的分区结果
        with open(f'./datasets/data/partition/RecEarser_{args.slices}.pk', 'rb') as f:
            C = pk.load(f)

    except IOError:
        # 如果预训练的分区结果不存在，则从头开始分区
        # 加载用户嵌入向量
        with open('./datasets/data/embeddings/user_embeds.pk', "rb") as f:
            uidW = pk.load(f)

        # 加载项目嵌入向量
        with open('./datasets/data/embeddings/item_embeds.pk', "rb") as f:
            iidW = pk.load(f)

        # 创建一个数据列表，包含所有可能的用户-项目对
        data = []
        # print(data.shape)
        # print(data.dtype)
        # print(data[:10])
        # print("最小值:", data.min())
        # print("最大值:", data.max())
        # print("平均值:", data.mean())
        # print("中位数:", np.median(data))
        # 生成所有可能的用户-项目对
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                data.append([i, j])

        # 计算每个分区中允许的最大用户-项目对数量
        max_number = 1.2 * (tensor.shape[0] * tensor.shape[1]) // args.slices
        # max_number = 1.2 * Tensor.shape[0] // args.slices

        # 随机选择初始分区中心
        center_id = random.sample(data, args.slices)
        center_user_value = []
        for i in range(args.slices):
            center_user_value.append([uidW[center_id[i][0]], iidW[center_id[i]][1]])

        print(center_id)  # 查看内容
        print([type(c) for c in center_id])  # 查看每个元素的类型

        C = None
        # 进行迭代优化分区
        for iterid in range(args.part_iter):
            C = [{} for _ in range(args.slices)] # 初始化分区容器
            C_number = [0 for _ in range(args.slices)] # 初始化分区计数器
            Scores = {}

            # 计算每个用户-项目对与各分区中心的距离
            for userid in trange(len(data), desc="Calculating Scores"):
                for sliceid in range(args.slices):
                    score_user = E_score2(np.array(uidW[data[userid][0]]), np.array(center_user_value[sliceid][0]))
                    score_item = E_score2(np.array(iidW[data[userid][1]]), np.array(center_user_value[sliceid][1]))

                    Scores[userid, sliceid] = - score_user * score_item

            # 根据得分排序，优先分配得分高的用户-项目对
            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

            visted = [False for _ in range(len(data))] # 记录已分配的用户-项目对
            for i in trange(len(Scores), desc="Assigning Pairs"):
                if not visted[Scores[i][0][0]]:
                    if C_number[Scores[i][0][1]] < max_number:
                        # 将用户-项目对分配到对应分区
                        if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]] = [
                                data[Scores[i][0][0]][1]]  # 把这个item放入对应的user元祖
                        else:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])

                        # print(C[Scores[i][0][1]][data[Scores[i][0][0]][0]])

                        visted[Scores[i][0][0]] = True
                        C_number[Scores[i][0][1]] += 1

            # 更新分区中心
            center_user_value_next = []
            for sliceid in trange(args.slices):
                temp_user_value = []
                temp_item_value = []
                user_mean, item_mean = None, None
                for userid in C[sliceid].keys():
                    for itemid in C[sliceid][userid]:
                        temp_user_value.append(uidW[userid])
                        temp_item_value.append(iidW[itemid])
                if len(temp_user_value):
                    user_mean = np.mean(temp_user_value)
                else:
                    user_mean = 0

                if len(temp_item_value):
                    item_mean = np.mean(temp_item_value)
                else:
                    item_mean = 0
                center_user_value_next.append([user_mean, item_mean])

            # 计算损失，监控优化进程
            loss = 0.0
            for sliceid in trange(args.slices):
                score_user = E_score2(np.array(center_user_value_next[sliceid][0]),
                                      np.array(center_user_value[sliceid][0]))
                score_item = E_score2(np.array(center_user_value_next[sliceid][1]),
                                      np.array(center_user_value[sliceid][1]))
                loss += (score_user * score_item)

            center_user_value = center_user_value_next
            log(f'iterid {iterid + 1} : loss = {loss:.30f}')
            for sliceid in range(args.slices):
                log(f'C[{sliceid}] number = {len(list(C[sliceid]))}')

        # 保存最终分区结果
        pk.dump(C, open(f'./datasets/data/partition/RecEarser_{args.slices}.pk', 'wb'))

    # 打印每个分区的用户-项目对数量
    print("每个分区的用户-项目对数量:")
    dic = {}
    for slice_id, users_items in C.items():
        dic[slice_id] = sum(len(items) for items in users_items.values())
    for slice_id, count in dic.items():
        print(f"分区 {slice_id}: {count} 个用户-项目对")

    # 初始化分割张量列表和分区标签字典
    split_Tensor = []
    # partition_labels = {}

    row_idx = [[] for _ in range(args.slices)]
    col_idx = [[] for _ in range(args.slices)]

    for sliceid in range(args.slices):
        temp = np.zeros_like(tensor) # 创建与原张量同形状的零张量

        # 遍历当前分区中的所有用户
        for userid in C[sliceid].keys():
            # 添加当前分区中所有用户-项目对的索引
            row_idx[sliceid] += [userid for _ in range(len(C[sliceid][userid]))]
            col_idx[sliceid] += [itemid for itemid in C[sliceid][userid]]

        # 根据索引填充分区张量的对应元素
        temp[row_idx[sliceid], col_idx[sliceid]] = tensor[row_idx[sliceid], col_idx[sliceid]]
        split_Tensor.append(temp)

        # 为当前分区打上标签，标签从1开始
        # partition_labels[sliceid + 1] = temp  # 使用sliceid + 1作为标签

    return split_Tensor



## 基于用户的平衡划分
# User-based Balanced Partition
def user_based_balanced_parition(tensor, args):
    # 将传入的训练数据赋值给data变量
    # data = tensor

    data = {}
    for k in range(len(tensor)):
        data[k] = tensor[k]  # 将每一行赋值给对应的key

    try:
        # 尝试加载预训练的分区结果
        with open(f'./datasets/data/partition/RecEarser_{args.slices}.pk', 'rb') as f:
            C = pk.load(f)

    except IOError:
        # 如果预训练的分区结果不存在，则从头开始分区
        # 加载用户嵌入向量
        with open('./datasets/data/embeddings/user_embeds.pk', "rb") as f:
            uidW = pk.load(f)

        # 加载项目嵌入向量
        # with open('./datasets/data/embeddings/item_embeds.pk', "rb") as f:
        #     iidW = pk.load(f)

        # get_data_interactions_1

        # Randomly select k centroids
        # 计算每个分区中允许的最大用户-项目对数量
        max_data = 1.2 * (tensor.shape[0] * tensor.shape[1]) // args.slices
        # max_data = 1.2 * len(data) / k
        # print data

        ## 随机选择初始质心
        # 从data的键（用户ID）中随机选择args.slices个作为初始质心
        centroids = random.sample(data.keys(), args.slices)
        # centroids = random.sample(data.keys(), k)
        # random_indices = np.random.choice(data.shape[0], args.slices, replace=False)
        # centroids = data[random_indices, :]

        # print(centroids)  # 查看内容
        # print([type(c) for c in centroids])  # 查看每个元素的类型

        # centro emb
        # print centroids

        ## 质心向量的计算
        # 为每个选定的质心（即用户ID），找到其对应的特征向量，形成初始质心向量列表centroembs
        centroembs = []
        for i in range(args.slices):
            temp_u = uidW[centroids[i]]
            centroembs.append(temp_u)

        C = None
        ## 迭代优化集群
        # 算法的核心！！！负责通过迭代优化过程来精细调整集群的划分
        for iterid in range(args.part_iter):
            C = [{} for _ in range(args.slices)] # 初始化C为k个空字典，每个字典代表一个集群，用于存储该集群内的用户数据
            Scores = {} # 初始化Scores字典，用于存储所有用户到每个质心的距离得分

            # 计算每个用户到每个质心的距离得分
            for userid in data.keys():
                for sliceid in range(args.slices):
                    # 计算每个用户到所有质心的距离
                    score_u = E_score2(np.array(uidW[userid]), np.array(centroembs[sliceid]))

                    Scores[userid, sliceid] = - score_u

            ## 更新集群分配
            # 将得分排序，以便将用户分配给使得分最小（即距离最近）的集群
            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

            fl = set() # 已分配的用户集合，用于确保每个用户只被分配到一个集群
            # visted = [False for _ in range(len(data))]  # 记录已分配的用户-项目对
            # 根据得分结果，分配用户到最近的集群，同时确保每个集群的数据量不超过max_data
            for i in trange(len(Scores)):
                if Scores[i][0][0] not in fl:
                    if len(C[Scores[i][0][1]]) < max_data:
                        C[Scores[i][0][1]][Scores[i][0][0]] = data[Scores[i][0][0]]
                        fl.add(Scores[i][0][0])

            ## 更新质心向量
            # 对于每个集群，计算其所有成员的特征向量的平均值，作为新的集群质心
            centroembs_next = []
            for sliceid in trange(args.slices):
                temp_u = []
                for userid in C[sliceid].keys():
                    temp_u.append(uidW[userid])
                if len(temp_u):
                    user_mean = np.mean(temp_u)
                else:
                    user_mean = 0
                centroembs_next.append([user_mean])

            # 初始化损失值
            loss = 0.0

            for sliceid in trange(args.slices):
                print(len(C[sliceid]))

            # 计算损失，这里的损失是指新旧质心向量之间的差异
            for sliceid in trange(args.slices):
                # score_u = E_score2(np.array(centroembs_next[sliceid],centroembs[sliceid]))
                score_u = E_score2(centroembs_next[sliceid], centroembs[sliceid])
                loss += score_u

            # 更新质心向量为新计算的向量
            centroembs = centroembs_next
            log(f'iterid {iterid + 1} : loss = {loss:.30f}')
            for sliceid in range(args.slices):
                log(f'C[{sliceid}] number = {len(list(C[sliceid]))}')

        # 保存最终分区结果
        pk.dump(C, open(f'datasets/data/partition/RecEarser_{args.slices}.pk', 'wb'))

    print("每个分区的用户-项目对数量:")
    dic = {}
    for slice_id in range(len(C)):  # 遍历C列表的索引
        users_items = C[slice_id]  # 获取当前索引下的分区数据（字典）
        dic[slice_id] = sum(len(items) for items in users_items.values())
    for slice_id, count in dic.items():
        print(f"分区 {slice_id}: {count}个用户-项目对")

    # 初始化分割张量列表和分区标签字典
    split_Tensor = []

    # 初始化用户和项目列表，用于存储每个集群的用户和项目数据
    users = [[] for _ in range(args.slices)]
    items = [[] for _ in range(args.slices)]

    # 填充用户和项目列表
    for sliceid in range(args.slices):
        temp = np.zeros_like(tensor)  # 创建与原张量同形状的零张量

        # 遍历当前分区中的所有用户
        for userid in C[sliceid].keys():
            # 添加当前分区中所有用户-项目对的索引
            users[sliceid] += [userid for _ in range(len(C[sliceid][userid]))]
            items[sliceid] += [itemid for itemid in C[sliceid][userid]]

        # 根据索引填充分区张量的对应元素
        temp[users[sliceid], items[sliceid]] = tensor[users[sliceid], items[sliceid]]
        split_Tensor.append(temp)

    # 返回集群的用户数据字典、用户列表和项目列表
    return split_Tensor
