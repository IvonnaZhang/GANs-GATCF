import torch as t
import numpy as np

from datasets.RecEraser import interaction_based_balanced_parition
from datasets.get_embedding import get_user_embedding, get_item_embedding
from datasets.packet import packet
from lib.parsers import get_parser
from datasets.dataset import *

if __name__ == '__main__':
    args = get_parser()
    # user_embedding = get_user_embedding(args)
    # item_embedding = get_item_embedding(args)

    # df = np.array(load_data(args))
    # dataset = ShardedTensorDataset(df, True, args)
    packet('./datasets/data/原始数据/userlist_table.csv', './datasets/data/原始数据/wslist_table.csv', './datasets/data/partition/RecEarser_5.pk')