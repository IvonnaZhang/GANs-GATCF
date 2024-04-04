import numpy as np
from torch.nn import *
import torch as t



def get_final_embedding():
    # 首先将其转换为NumPy数组
    with open(f'../datasets/data/partition/sub/updated_user_embeddings.pk', 'wb') as file:
        agg_user_embeds = pickle.load(file)
    with open(f'../datasets/data/partition/sub/updated_item_embeddings.pk', 'wb') as file:
        agg_item_embeds = pickle.load(file)

    combined_embeddings_array = np.array(agg_user_embeds)
    # 获取按照用户ID排序的索引
    sorted_indices = np.argsort(combined_embeddings_array[:, 0])
    # 使用这些索引来对整个矩阵排序
    final_user_embeds = combined_embeddings_array[sorted_indices]
    agg_item_embeds_tensor = t.stack(agg_item_embeds)  # 大小为 [N, 3525, 128]
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

