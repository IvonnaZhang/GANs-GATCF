# coding : utf-8
# Author : yuxiang Zeng

from datasets.data_generator import get_train_valid_test_dataset
from lib.load_dataset import get_pytorch_index
from utils.dataloader import get_dataloaders
import torch
from torch.utils.data import Dataset

from utils.logger import Logger


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        # Setup Logger
        log = Logger(args)
        args.log = log

        # 加载原始数据
        # Step1：用load_data加载数据data
        self.data = exper_type.load_data(args)

        # 预处理数据（删）
        # Step2：用preprocess_data对data进行预处理得到data
        self.data = exper_type.preprocess_data(self.data, args)

        # 切分训练测试
        # Step3：get_train_valid_test_dataset通过data得到train_tensor
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = get_train_valid_test_dataset(self.data, args)

        # 装载数据
        # Step4：TensorDataset处理train_tensor得到train_set
        self.train_set, self.valid_set, self.test_set = TensorDataset(self.train_tensor), TensorDataset(self.valid_tensor), TensorDataset(self.test_tensor)

        # 装进pytorch
        # Step5： get_dataloaders处理train_set得到train_loader
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        # 基本信息
        args.log.only_print(f'Train_length : {len(self.train_loader) * args.bs} Valid_length : {len(self.valid_loader) * args.bs * 16} Test_length : {len(self.test_loader) * args.bs * 16}')

    def get_tensor(self):
        return self.train_tensor, self.valid_tensor, self.test_tensor

    def trainLoader(self):
        return self.train_loader

    def validLoader(self):
        return self.valid_loader

    def testLoader(self):
        return self.test_loader

    def fullLoader(self):
        return self.fullloader


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor):
        self.tensor = tensor
        self.indices = get_pytorch_index(tensor)
        self.indices = self.delete_zero_row(self.indices)

    def __getitem__(self, idx):
        output = self.indices[idx, :-1]  # 去掉最后一列
        inputs = tuple(torch.as_tensor(output[i]).long() for i in range(output.shape[0]))
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]


    @staticmethod
    def delete_zero_row(tensor):
        row_sums = tensor.sum(axis=1)
        nonzero_rows = (row_sums != 0).nonzero().squeeze()
        filtered_tensor = tensor[nonzero_rows]
        return filtered_tensor
