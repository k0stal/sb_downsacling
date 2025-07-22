import os
import argparse
import torch
import torch.nn as nn

from src.trainer import Trainer, Evaluator
from src.data_loader import get_data
from src.models.srcnn import SRCNN
from src.models.espcn import ESPCN
from src.models.bicubic import Bicubic
from src.models.fno import FNO

MODEL_LIST = {
    'Bicubic': lambda args: Bicubic(args.upscale_factor),
    'SRCNN': lambda args: SRCNN(args.n_channels, args.upscale_factor),
    'ESPCN': lambda args: ESPCN(args.n_channels, args.upscale_factor),
    'FNO': lambda args: FNO(args.n_channels, args.upscale_factor, args.layers, args.width, args.modes, args.modes)
}

parser = argparse.ArgumentParser(description='Training parameters')

# Dataset arguments
parser.add_argument('--data_path', type=str, default='/mnt/personal/kostape4/downscale/data/climate', help='The folder path of dataset')
parser.add_argument('--crop_size', type=int, default=0, help='Crop size for high-resolution snapshots')
parser.add_argument('--n_patches', type=int, default=8, help='Number of patches')

# Training arguments
parser.add_argument('--model', type=str, default='subpixelCNN', help='Model')
parser.add_argument('--epochs', type=int, default=300, help='Max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='Computing device')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--logdir', type=str, default='logs', help='Directory for logs')
parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='Type of scheduler')
parser.add_argument('--evaluate', type=bool, default=False, help='Evaluate the given model')

# Model-specific arguments
parser.add_argument('--n_channels', type=int, default=1, help='Number of channels to process')
parser.add_argument('--upscale_factor', type=int, default=8, help='Upscale factor')

# FNO-specific arguments
parser.add_argument('--modes', type=int, default=12, help='FNO modes')
parser.add_argument('--width', type=int, default=32, help='FNO width')
parser.add_argument('--layers', type=int, default=4, help='FNO layers')


def train_model(model: nn.Module, args: argparse.Namespace) -> None:
    train_loader, val_loader = get_data(args)
    trainer = Trainer(args, model, train_loader, val_loader)
    trainer.train()


def evalulate_mode(model: nn.Module, args: argparse.Namespace) -> None:
    test_loader = get_data(args)
    evaluator = Evaluator(args, model, test_loader)
    evaluator.test()


def main(args: argparse.Namespace) -> None:
    print(f'Training setting: {args}')

    # Ensure reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    model = MODEL_LIST[args.model](args).to(args.device)

    model = nn.DataParallel(model)

    if not args.evaluate:
        train_model(model, args)
    else:
        evalulate_mode(model, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

