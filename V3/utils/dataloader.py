# coding : utf-8
# Author : yuxiang Zeng
import platform
import multiprocessing

from torch.utils.data import DataLoader

def get_dataloaders(train_set, valid_set, test_set, args):
    # max_workers = multiprocessing.cpu_count()
    max_workers = 8

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        # num_workers=max_workers if platform.system() == 'Linux' else 0,
        # prefetch_factor=4 if platform.system() == 'Linux' else 2
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.bs * 16,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=max_workers if platform.system() == 'Linux' else 0,
        # prefetch_factor=4 if platform.system() == 'Linux' else 2
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs * 16,  # 8192
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=max_workers if platform.system() == 'Linux' else 0,
        # prefetch_factor=4 if platform.system() == 'Linux' else 2
    )

    return train_loader, valid_loader, test_loader
