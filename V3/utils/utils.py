# Author : yuxiang Zeng
# 损失函数，反馈日志
import time
import nbformat
import numpy as np
import pandas as pd
import torch as t
import random
import csv
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_settings(args, config):
    # args.dimension = int(config[args.interaction]['dimension'])
    # args.order = int(config[args.interaction]['order'])
    # args.slice_epochs = int(config[args.interaction]['slice_epochs'])
    # # args.verbose = int(config['Experiment']['verbose'])
    # args.external_dim = int(config[args.interaction]['external_dim'])
    # args.att_lr = float(config[args.interaction]['att_lr'])
    # args.att_decay = float(config[args.interaction]['att_decay'])
    args.n_clusters = args.slices

    if args.part_type == 1:
        args.agg_type = config['SISA']['agg_type']
        args.slice_epochs = 35

    elif args.part_type == 3:
        args.agg_type = config['RecEraser']['agg_type']

    if args.retrain:
        args.slice_lr = 1e-3
        args.slice_decay = 0.
        args.verbose = 1
        args.slices = 1

    if not args.verbose:
        args.test = False

    if args.debug:
        # args.slice_epochs = 1
        # args.agg_epochs = 1
        # args.record = 0
        args.rounds = 2
        args.epochs = 1
        args.record = 1
        args.lr = 1e-3
        args.decay = 1e-3

    if args.experiment:
        args.record = 1
        args.program_test = 0
        args.verbose = 10

    if args.interaction == 'NeuCF':
        args.slice_lr = 0.004
        args.agg_lr = 0.001
    elif args.interaction == 'CSMF':
        args.slice_lr = 0.01
        args.agg_lr = 0.008
    elif args.interaction == 'MF':
        args.slice_lr = 0.008
        args.agg_lr = 0.004
    elif args.interaction == 'GraphMF':
        args.slice_lr = 0.008
        args.agg_lr = 0.001

    return args


# 存储全部文件地址
class File_address:
    def __init__(self, args):
        self.log = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/' + str(f'{args.density:.2f}') + '.日志'

        self.result_dir = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/metrics'
        self.time_dir = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/time'

        self.Final_result_density_txt = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/metrics' + '/Final_result_density_' + str(args.dataset) + '_' + str(f'{args.density:.2f}') + '.txt'
        self.Final_result_density_csv = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/metrics' + '/Final_result_density_' + str(args.dataset) + '_' + str(f'{args.density:.2f}') + '.csv'
        self.Training_result_density_txt = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/metrics' + '/Training_result_density_' + str(args.dataset) + '_' + str(f'{args.density:.2f}') + '.txt'

        self.Final_time_density_txt = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/time' + '/Final_time_density_' + str(args.dataset) + '_' + str(f'{args.density:.2f}') + '.txt'
        self.Final_time_density_csv = './Result/' + args.interaction + '/part_type_' + str(args.part_type) + '/' + str(args.dataset) + '/slices_' + str(args.slices) + '/time' + '/Final_time_density_' + str(args.dataset) + '_' + str(f'{args.density:.2f}') + '.csv'


def debug(data, string):
    print(string, '数据类型', type(data))
    print(data)


def makedir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    return False


# 记录epoch，查看是否还有进步空间
def per_epoch_result_start(args):
    file_address = File_address(args)
    file = file_address.Training_result_density_txt
    makedir(file_address.result_dir)

    with open(file, 'w') as f:
        f.write("Experiment results are as follows:\n")


def per_epoch_in_txt(args, epoch, MAE, RMSE, NMAE, MRE, NPRE, train_time, early_stop):
    MAE, RMSE, NMAE, MRE, NPRE = np.array(MAE), np.array(RMSE), np.array(NMAE), np.array(MRE), np.array(NPRE)
    file_address = File_address(args)
    file = file_address.Training_result_density_txt
    makedir(file_address.result_dir)

    PRINT_ROUND = f'Epoch : {epoch:2d} result: MAE = {MAE:.3f}, RMSE = {RMSE:.3f}, NMAE = {NMAE:.3f}, MRE = {MRE:.3f}, NPRE = {NPRE:.3f}  train_time = {train_time :.2f} s'

    if epoch == args.agg_epochs or early_stop or epoch == args.slice_epochs:
        PRINT_ROUND += '\n'
    with open(file, mode='a') as f:
        f.write(PRINT_ROUND + '\n')



# 记录每一轮的结果
def per_round_result_start(args):
    file_address = File_address(args)
    file = file_address.Final_result_density_txt
    makedir(file_address.result_dir)

    with open(file, 'w') as f:
        f.write("Experiment results are as follow:\n\n")


def per_round_result_in_txt(args, Round, MAE, RMSE, NMAE, MRE, NPRE):
    MAE, RMSE, NMAE, MRE, NPRE = np.array(MAE), np.array(RMSE), np.array(NMAE), np.array(MRE), np.array(NPRE)
    file_address = File_address(args)
    file = file_address.Final_result_density_txt
    makedir(file_address.result_dir)

    with open(file, mode='a') as f:
        PRINT_FINAL = f'Experiment {Round:2d} : MAE = {MAE:.3f}, RMSE = {RMSE:.3f}, NMAE = {NMAE:.3f}, MRE = {MRE:.3f}, NPRE = {NPRE:.3f}'
        f.write(PRINT_FINAL + '\n')


def final_result_in_txt(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE):
    RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE = np.array(RunMAE), np.array(RunRMSE), np.array(RunNMAE), np.array(RunMRE), np.array(RunNPRE)
    file_address = File_address(args)
    file = file_address.Final_result_density_txt
    makedir(file_address.result_dir)

    with open(file, 'a') as f:
        PRINT_ROUND = f'Final result: MAE = {np.mean(RunMAE, axis = 0):.3f}, RMSE = {np.mean(RunRMSE, axis = 0):.3f}, NMAE = {np.mean(RunNMAE, axis = 0):.3f}, MRE = {np.mean(RunMRE, axis = 0):.3f}, NPRE = {np.mean(RunNPRE, axis = 0):.3f}\n'
        f.write('\n' + PRINT_ROUND + '\n')


# 记录训练时间
def per_slice_time_start(args):
    file_address = File_address(args)
    file = file_address.Final_time_density_txt
    makedir(file_address.time_dir)

    with open(file, 'w') as f:
        f.write("Experiment time result are as follow:\n\n")


def per_slice_time_in_txt(args, round, sliceId, training_time):
    file_address = File_address(args)
    file = file_address.Final_time_density_txt
    makedir(file_address.time_dir)

    with open(file, mode='a') as f:
        PRINT_FINAL = f'Experiment {round:2d} : Slice {sliceId:2d} training time = {training_time:.2f} s'
        f.write(PRINT_FINAL + '\n')


def per_round_agg_time_in_txt(args, round, training_time):
    file_address = File_address(args)
    file = file_address.Final_time_density_txt
    makedir(file_address.time_dir)

    with open(file, mode='a') as f :
        PRINT_FINAL = f'Experiment {round:2d} : Aggregators training time = {training_time:.2f} s\n'
        f.write(PRINT_FINAL + '\n')


# CSV
def per_round_result_start_csv(args):
    file_address = File_address(args)
    file = file_address.Final_result_density_csv
    makedir(file_address.result_dir)

    with open(file, mode='w', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = ('ROUND', 'MAE', 'RMSE', 'NMAE', 'MRE', 'NPRE')
        csv_write.writerow(text)


def per_round_result_in_csv(args, round, MAE, RMSE, NMAE, MRE, NPRE):
    MAE, RMSE, NMAE, MRE, NPRE = np.array(MAE), np.array(RMSE), np.array(NMAE), np.array(MRE), np.array(NPRE)
    file_address = File_address(args)
    file = file_address.Final_result_density_csv
    makedir(file_address.result_dir)

    with open(file, mode='a', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = round, MAE, RMSE, NMAE, MRE, NPRE
        csv_write.writerow(text)
    f.close()


def final_result_in_csv(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE):
    RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE = np.array(RunMAE), np.array(RunRMSE), np.array(RunNMAE), np.array(RunMRE), np.array(RunNPRE)
    file_address = File_address(args)
    file = file_address.Final_result_density_csv
    makedir(file_address.result_dir)

    with open(file, mode='a', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = 'Final', np.mean(RunMAE, axis = 0), np.mean(RunRMSE, axis = 0), np.mean(RunNMAE, axis = 0), np.mean(RunMRE, axis = 0), np.mean(RunNPRE, axis = 0)
        csv_write.writerow(text)


def per_round_time_start_csv(args):
    file_address = File_address(args)
    file = file_address.Final_time_density_csv
    makedir(file_address.time_dir)

    with open(file, mode='w', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = ('ROUND', 'SLICES', 'training_time')
        csv_write.writerow(text)


def per_slice_time_in_csv(args, round, sliceId, training_time):
    file_address = File_address(args)
    file = file_address.Final_time_density_csv
    makedir(file_address.time_dir)

    with open(file, mode='a', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = round, sliceId, training_time
        csv_write.writerow(text)


def per_round_agg_time_in_csv(args, round, training_time):
    file_address = File_address(args)
    file = file_address.Final_time_density_csv
    makedir(file_address.time_dir)

    with open(file, mode='a', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        text = round, 'Aggregators', training_time
        csv_write.writerow(text)


# 将已经训练结果先加进来
def trained(args):
    file_address = File_address(args)
    file = file_address.Final_result_density_csv

    df = np.array(pd.read_csv(file))
    df = df[:, 1:]
    df.astype('float')
    return df[:, 0], df[:, 1], df[:, 2], df[:, 3], df[:, 4]


# 日志记录
def log(string):
    import time
    if string[0] == '\n':
        print('\n', end = '')
        string = string[1:]
    print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())), string)


def computer_info():
    import platform

    def showinfo(tip, info):
        print("{} : {}".format(tip, info))

    showinfo("操作系统及版本信息", platform.platform())
    showinfo('获取系统版本号', platform.version())
    showinfo('获取系统名称', platform.system())
    showinfo('系统位数', platform.architecture())
    showinfo('计算机类型', platform.machine())
    showinfo('计算机名称', platform.node())
    showinfo('处理器类型', platform.processor())
    showinfo('计算机相关信息', platform.uname())

def to_cuda(inputs, values, args):
    import torch
    inputs = [tensor.to(args.device) for tensor in inputs]
    values = values.to(torch.float32).to(args.device)
    return inputs, values

def optimizer_zero_grad(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()

def optimizer_step(*optimizers):
    for optimizer in optimizers:
        optimizer.step()

def lr_scheduler_step(*lr_scheduler):
    for scheduler in lr_scheduler:
        scheduler.step()

#########################################################################
def create_ipynb_file(cells, file_name):
    # 创建一个空白的Notebook
    nb = nbformat.v4.new_notebook()
    # 添加单元格内容
    for cell in cells:
        cell_type = cell.get('cell_type', 'code')
        source = cell.get('source', '')
        metadata = cell.get('metadata', {})

        if cell_type == 'code':
            nb.cells.append(nbformat.v4.new_code_cell(source=source, metadata=metadata))
        elif cell_type == 'markdown':
            nb.cells.append(nbformat.v4.new_markdown_cell(source=source, metadata=metadata))
        else:
            raise ValueError("Invalid cell type: {}".format(cell_type))
    # 检查文件是否存在并添加编号
    file_path = f'{time.localtime(time.time()).tm_mon}.{time.localtime(time.time()).tm_mday} '
    file_path += f'{file_name}.ipynb'  # 指定文件路径和名称
    i = 2
    while os.path.exists(file_path):
        file_path = f'{time.localtime(time.time()).tm_mon}.{time.localtime(time.time()).tm_mday} '
        file_path += f'{file_name}{i}.ipynb'
        i += 1
    # 保存为.ipynb文件
    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())) + f'\"{file_path}\"' + ' 文件保存成功!')



def create_sh_file(cells, file_name):
    # 检查文件是否存在并添加编号
    file_path = f'{time.localtime(time.time()).tm_mon}.{time.localtime(time.time()).tm_mday} '
    file_path += f'{file_name}.sh'  # 指定文件路径和名称
    i = 2
    while os.path.exists(file_path):
        file_path = f'{time.localtime(time.time()).tm_mon}.{time.localtime(time.time()).tm_mday} '
        file_path += f'{file_name}{i}.sh'
        i += 1
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in range(len(cells)):
            f.write(cells[item])
            f.write('\n')
    print(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime(time.time())) + f'\"{file_path}\"' + ' 文件保存成功!')
