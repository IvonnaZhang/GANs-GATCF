# coding : utf-8
# Author : yuxiang Zeng
import numpy as np


# 一定密度下的采样切分数据集
def get_train_valid_test_dataset(tensor, args):
    # 假设这里的tensor不再需要，因为我们直接从文件中加载数据
    # quantile 计算可能仍然需要，根据实际逻辑决定
    quantile = np.percentile(tensor, q=100)
    # tensor[tensor > quantile] = 0
    # tensor /= quantile
    # density = args.density
    #
    # mask = np.random.rand(*tensor.shape).astype('float32')  # [0, 1]
    #
    # mask[mask > density] = 1
    # mask[mask < density] = 0
    #
    # train_Tensor = tensor * (1 - mask)
    #
    # # size = int(0.05 * np.prod(tensor.shape))
    # size = 0
    #
    # trIdx, tcIdx = mask.nonzero()
    # p = np.random.permutation(len(trIdx))
    # trIdx, tcIdx = trIdx[p], tcIdx[p]
    #
    # vrIdx, vcIdx = trIdx[:size], tcIdx[:size]
    # trIdx, tcIdx = trIdx[size:], tcIdx[size:]
    #
    # valid_Tensor = np.zeros_like(tensor)
    # test_Tensor = np.zeros_like(tensor)
    #
    # valid_Tensor[vrIdx, vcIdx] = tensor[vrIdx, vcIdx]
    #
    # test_Tensor[trIdx, tcIdx] = tensor[trIdx, tcIdx]

    # 加载数据集
    train_Tensor = np.loadtxt('./datasets/gans_data/train_matrix_生成.txt', dtype='float32')
    valid_Tensor = np.loadtxt('./datasets/gans_data/valid_matrix.txt', dtype='float32')
    test_Tensor = np.loadtxt('./datasets/gans_data/test_matrix.txt', dtype='float32')

    # 如果原始数据集tensor用于其他目的，例如计算quantile，应保留其原始加载逻辑
    # 否则，如果quantile仅用于归一化，则可能需要调整此逻辑，以适应新的数据

    return train_Tensor, valid_Tensor, test_Tensor, quantile
