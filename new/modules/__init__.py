from modules.GATCF import GATCF
from modules.edge_train import EdgeModel


def get_model(datasets, args):
    user_num, serv_num = datasets.data.shape
    # print(user_num, serv_num)
    if args.model == 'GATCF':
        return GATCF(user_num, serv_num, args)
    if args.model == "EdgeModel":
        return EdgeModel(user_num, serv_num, args)
    else:
        raise NotImplementedError
