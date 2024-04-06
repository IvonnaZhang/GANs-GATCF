import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()
        self.feature_dim = feature_dim
        self.scale = 1.0 / (self.feature_dim ** 0.5)
        self.query = nn.Linear(self.feature_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.value = nn.Linear(self.feature_dim, self.feature_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output



# # 检查结果的尺寸
# print(averaged_result.shape)

