# coding : utf-8
# Author : yuxiang Zeng

import torch as t
import numpy as np


# 精度计算
def error_metrics(real_vec, esti_vec):
    if isinstance(real_vec, np.ndarray):
        real_vec = real_vec.astype(float)
    elif isinstance(real_vec, t.Tensor):
        real_vec = real_vec.cpu().detach().numpy().astype(float)
    if isinstance(esti_vec, np.ndarray):
        esti_vec = esti_vec.astype(float)
    elif isinstance(esti_vec, t.Tensor):
        esti_vec = esti_vec.cpu().detach().numpy().astype(float)

    absError = np.abs(esti_vec - real_vec)
    MAE = np.mean(absError)
    RMSE = np.linalg.norm(absError) / np.sqrt(np.array(absError.shape[0]))
    NMAE = np.sum(np.abs(real_vec - esti_vec)) / np.sum(real_vec)
    relativeError = absError / real_vec
    NRMSE = np.sqrt(np.sum((real_vec - esti_vec) ** 2)) / np.sqrt(np.sum(real_vec ** 2))
    NPRE = np.array(np.percentile(relativeError, 90))  #

    return {
        'MAE': MAE,
        'RMSE': RMSE,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'NPRE': NPRE,
    }
