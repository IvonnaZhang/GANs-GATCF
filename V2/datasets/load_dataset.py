# coding : utf-8
# Author : yuxiang Zeng
import pickle
import torch
import numpy as np

def get_exper(args):
    return experiment1(args)


class experiment1:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        string = args.path + '/' + args.dataset + 'Matrix' + '.txt'
        tensor = np.loadtxt(open(string, 'rb'))
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data


def get_pytorch_index(data):
    userIdx, servIdx = data.nonzero()
    values = data[userIdx, servIdx]
    idx = torch.as_tensor(np.vstack([userIdx, servIdx, values]).T)
    return idx







