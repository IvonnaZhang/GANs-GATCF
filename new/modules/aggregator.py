import pickle

import numpy as np
from torch.nn import *
import torch as t


def get_final_embedding():
    with open(f'../datasets/data/partition/sub/updated_user_embeddings.pk', 'wb') as file:
        agg_user_embeds = pickle.load(file)
    with open(f'../datasets/data/partition/sub/updated_item_embeddings.pk', 'wb') as file:
        agg_item_embeds = pickle.load(file)
    combined_embeddings_array = np.array(agg_user_embeds)
    # 获取按照用户ID排序的索引
    sorted_indices = np.argsort(combined_embeddings_array[:, 0])
    # 使用这些索引来对整个矩阵排序
    final_user_embeds = combined_embeddings_array[sorted_indices]
    agg_item_embeds_tensor = t.stack(agg_item_embeds)
    multihead_attn = MultiheadAttention(embed_dim=128, num_heads=2)
    query = key = value = agg_item_embeds_tensor
    # 注意力层的输出大小也会是 [seq_len, batch_size, embed_dim]
    attn_output, _ = multihead_attn(query, key, value)
    # 简单地取平均
    final_serv_embeds = attn_output.mean(dim=1)
    return final_user_embeds, final_serv_embeds

