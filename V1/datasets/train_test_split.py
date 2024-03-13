# coding : utf-8
# Author : yuxiang Zeng

import numpy as np

def get_train_valid_test_dataset(tensor, args):
    quantile = np.percentile(tensor, q=100)
    # tensor[tensor > quantile] = 0

    tensor = tensor / (np.max(tensor))  # 如果数据有分布偏移，记得处理数据

    trainsize = int(np.prod(tensor.size) * args.density)
    validsize = int((np.prod(tensor.size)) * 0.05) if args.valid else int((np.prod(tensor.size) - trainsize) * 1.0)

    rowIdx, colIdx = tensor.nonzero()
    p = np.random.permutation(len(rowIdx))
    rowIdx, colIdx = rowIdx[p], colIdx[p]

    trainRowIndex = rowIdx[:trainsize]
    trainColIndex = colIdx[:trainsize]

    traintensor = np.zeros_like(tensor)
    traintensor[trainRowIndex, trainColIndex] = tensor[trainRowIndex, trainColIndex]

    validStart = trainsize
    validRowIndex = rowIdx[validStart:validStart + validsize]
    validColIndex = colIdx[validStart:validStart + validsize]
    validtensor = np.zeros_like(tensor)
    validtensor[validRowIndex, validColIndex] = tensor[validRowIndex, validColIndex]

    testStart = validStart + validsize
    testRowIndex = rowIdx[testStart:]
    testColIndex = colIdx[testStart:]
    testtensor = np.zeros_like(tensor)
    testtensor[testRowIndex, testColIndex] = tensor[testRowIndex, testColIndex]

    return traintensor, validtensor, testtensor, quantile

