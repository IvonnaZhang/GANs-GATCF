# coding : utf-8
# Author : yuxiang Zeng
import configparser
import os
import time
import collections
import numpy as np

from lib.parsers import get_parser
from modules import get_model
from lib.load_dataset import get_exper
from utils.datamodule import DataModule
from utils.logger import Logger
from utils.monitor import EarlyStopping
from utils.utils import set_seed, set_settings


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

global log


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = get_exper(args)
    dataModule = DataModule(exper, args)

    model = get_model(dataModule, args)

    monitor = EarlyStopping(args.patience)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = dataModule.max_value
    train_time = []
    for epoch in range(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(dataModule)
        valid_error = model.valid_one_epoch(dataModule)
        monitor.track(epoch, model.state_dict(), valid_error['MAE'])
        train_time.append(time_cost)

        if args.verbose and epoch % args.verbose == 0:
            log.only_print(
                f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")

        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)

    sum_time = sum(train_time[: monitor.best_epoch])

    results = model.test_one_epoch(dataModule) if args.valid else valid_error

    log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s\n')

    return {
        'MAE': results["MAE"],
        'RMSE': results["RMSE"],
        'NMAE': results["NMAE"],
        'NRMSE': results["NRMSE"],
        'TIME': sum_time,
    }


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])

    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


if __name__ == '__main__':
    args = get_parser()
    config = configparser.ConfigParser()
    set_settings(args, config)

    # Setup Logger
    log = Logger(args)
    args.log = log

    # Record Experiments Config
    log(str(args))

    # Run Experiments
    RunExperiments(log, args)
