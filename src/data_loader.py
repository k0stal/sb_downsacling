import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset.dataset import SuperBench

def get_data(args: argparse.Namespace) -> tuple[DataLoader, DataLoader] | DataLoader:
    if not args.evaluate:
        train_path = f'{args.data_path}/train'
        # using only validation data for extrapolation
        val_path = f'{args.data_path}/valid_2'
        train_loader = DataLoader(SuperBench(args, train_path))
        val_loader = DataLoader(SuperBench(args, val_path, False))
        return train_loader, val_loader
    else:
        # using only test data for extrapolation
        test_path = f'{args.data_path}/test_2'
        test_loader = DataLoader(SuperBench(test_path, args.crop_size, args.n_patches))
        return test_loader
    print('Dataset loaded...')

