import torch as t
import numpy as np

from datasets.packet import packet
from lib.parsers import get_parser
from modules.get_embedding import get_user_embedding, get_item_embedding

if __name__ == '__main__':
    args = get_parser()
    user_embedding = get_user_embedding(args, './datasets/data/partition/userlist_group_5.csv', 5)
    # item_embedding = get_item_embedding(args, './datasets/data/partition/wslist_group_1.csv', 1)

    # df = np.array(load_data(args))
    # dataset = ShardedTensorDataset(df, True, args)
    # packet('./datasets/data/原始数据/userlist_table.csv', './datasets/data/原始数据/wslist_table.csv', './datasets/data/partition/RecEarser_5.pk')