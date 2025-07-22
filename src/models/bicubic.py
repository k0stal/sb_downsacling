import torch

class Bicubic(torch.nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upsacle_factor = upscale_factor
        
    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, 
            scale_factor=self.upscale_factor, 
            mode='bicubic', 
            align_corners=False,
            antialias=True
        )
        
        return x
