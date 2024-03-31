import numpy as np
from torch.nn import *
import torch as t



def get_final_embedding(agg_user_embeds,agg_item_embeds):
    # 首先将其转换为NumPy数组
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
#突然发现好像不用那么麻烦




# # 初始化模型
#
# final_user_embeds, final_serv_embeds = get_final__embedding()
#
# class FinalMLP(nn.Module):
#     def __init__(self, args,final_user_embeds, final_serv_embeds ):
#         super(FinalMLP, self).__init__(args,final_user_embeds, final_serv_embeds )
#         self.args = args
#         self.final_user_embeds = final_user_embeds
#         self.final_serv_embeds = final_serv_embeds
#         self.user_attention = SpGAT(self.final_user_embeds, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
#         self.item_attention = SpGAT(self.final_serv_embeds, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
#         self.layers = torch.nn.Sequential(
#             torch.nn.Linear(2 * args.dimension, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128),
#             torch.nn.LayerNorm(128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 1)
#         )
#
#
#     def forward(self, final_user_embeds, final_serv_embeds):
#         estimated = self.layers(torch.cat((final_user_embeds, final_serv_embeds ), dim=-1)).sigmoid().reshape(-1)
#         return estimated
#
#
#
# # 创建MLP模型
# mlp_model = FinalMLP(args,final_user_embeds, final_serv_embeds )
#
# # 经过MLP处理
# output = mlp_model(args,final_user_embeds, final_serv_embeds )
