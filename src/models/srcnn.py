import torch
import torch.nn as nn
import torch.nn.functional

class SRCNN(nn.Module):
    def __init__(self, n_channels: int, upscale_factor: int) -> None:
        super().__init__()

        self.upscale_factor = upscale_factor
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=n_channels, kernel_size=5, padding=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(
            input=x, 
            scale_factor=self.upscale_factor,
            mode='bicubic', 
            align_corners=False,
            antialias=True
        )
        return self.model(x)
