import numpy as np
import h5py
import torch
import glob
import argparse
import os
import xarray as xr

from torchvision.transforms.v2 import RandomCrop
from torch.utils.data import Dataset

class SuperBench(Dataset):
    """
    Dataloader class for SuperBench weather dataset.
    """

    def __init__(self, args: argparse.Namespace, file_path: str, train: bool = True) -> None:
        self.upscale_factor = args.upscale_factor
        self.n_patches = args.n_patches
        self.n_channels = args.n_channels
        self.crop_size = args.crop_size
        self.train = train
        self.random_crop = RandomCrop(size=self.crop_size)

        h5_files_paths = sorted(glob.glob(file_path + "/*.h5"))
        if not h5_files_paths:
            raise FileNotFoundError(f"No .h5 files found in {file_path}")

        data = []

        print(f"Loading data from h5 files...")
        for file in h5_files_paths:
            with h5py.File(file, "r") as f:
                dataset = f["fields"]
                # read all timestamp but only the second channel
                # that is the 2m temperature
                # TODO verify that the second channel is the 2m temperature
                t2m = dataset[:, 1:2]
                t2m = torch.from_numpy(t2m)
                data.append(t2m)

        self.Y = torch.cat(data)
        
        # validation data doesn't match exactly the training ones
        self.Y = self.Y[:, :, :-1]
        
        print(f"{'Training' if self.train else 'Validation'} data loaded.")

    def upscale(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            y,
            scale_factor=self.upscale_factor**-1,
            mode="bicubic",
            antialias=True,
        )
        return x
def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: y = self.Y[index]
        # if testing process data with original size
        if self.train and self.crop_size != 0:
            y = self.random_crop(y)
        # torch.nn.functional.interpolate requires a 4-dimensional tensor
        x = self.upscale(y.unsqueeze(0)).squeeze(0)
        return x, y

    def __len__(self) -> int:
        return len(self.Y)

class Cerra(Dataset):
    """
    Dataloader class for CERRA 2m temperature dataset stored in GRIB format.
    """

    def __init__(self, args: argparse.Namespace, file_path: str, train: bool = True) -> None:
        self.upscale_factor = args.upscale_factor
        self.n_patches = args.n_patches
        self.n_channels = args.n_channels
        self.crop_size = args.crop_size
        self.train = train
        self.random_crop = RandomCrop(size=self.crop_size)

        grib_files = sorted(glob.glob(os.path.join(file_path, "*.grib")))
        if not grib_files:
            raise FileNotFoundError(f"No .grib files found in {file_path}")

        data = []

        print(f"Loading data from GRIB files...")
        for file in grib_files:
            ds = xr.open_dataset(file, engine="cfgrib")
            if "2t" not in ds:
                print(f"Warning: '2t' variable not found in {file}, skipping.")
                continue

            t2m = ds["2t"].values  # shape: (time, lat, lon)
            t2m = torch.from_numpy(t2m).float()  # Convert to float32 tensor

            # Add channel dimension: (time, 1, H, W)
            t2m = t2m.unsqueeze(1)
            data.append(t2m)

        if not data:
            raise RuntimeError("No valid GRIB files with '2t' variable were loaded.")

        self.Y = torch.cat(data, dim=0)  # shape: (N, 1, H, W)

        print(f"{'Training' if self.train else 'Validation'} data loaded. Shape: {self.Y.shape}")

    def upscale(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            y,
            scale_factor=self.upscale_factor**-1,
            mode="bicubic",
            antialias=True,
        )
        return x

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.Y[index]
        if self.train and self.crop_size != 0:
            y = self.random_crop(y)
        x = self.upscale(y.unsqueeze(0)).squeeze(0)
        return x, y

    def __len__(self) -> int:
        return len(self.Y)

