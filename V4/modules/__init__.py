from modules.GATCF import GATCF


def get_model(datasets, args):
    user_num, serv_num = datasets.data.shape
    # print(user_num, serv_num)
    if args.model == 'GATCF':
        return GATCF(user_num, serv_num, args)
    else:
        raise NotImplementedError
