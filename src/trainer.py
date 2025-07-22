import os
import argparse
import torch
import torchmetrics
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

METRICS = torchmetrics.MetricCollection({
    "mae": torchmetrics.MeanAbsoluteError(),
    "rmse": torchmetrics.MeanSquaredError(squared=False),
    "psnr": torchmetrics.image.PeakSignalNoiseRatio(),
    "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(),
})

class Trainer:
    """Handles training and validation of the model."""

    def __init__(self, args: argparse.Namespace, model: nn.Module, train_loader, val_loader):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = nn.MSELoss()
        self.train_metrics = METRICS.clone().to(self.device)
        self.validation_metrics = METRICS.clone().to(self.device)
        
        # might AdamW be better?
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.best_val_loss = np.inf

        # erase nbn
        if args.model != 'FNO':
            self.model_path = f"{args.model}_{args.upscale_factor}_{args.crop_size}_{args.epochs}_{args.lr}"
        else:
            self.model_path = f"{args.model}_{args.upscale_factor}_{args.crop_size}_{args.epochs}_{args.lr}_{args.modes}_{args.width}_{args.layers}_nbn"

        log_dir = os.path.join(args.logdir, self.model_path) 
        self.writer = SummaryWriter(log_dir=log_dir)

    def train(self) -> None:
        print("Starting Training...")
        for epoch in range(self.args.epochs):
            self.model.train()
            self.train_metrics.reset()
            train_loss = 0

            for data, target in tqdm(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)
                loss = self.loss(output, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                self.train_metrics.update(output, target)

            train_loss /= len(self.train_loader)
            train_metrics_res = self.train_metrics.compute()
            val_loss, val_metrics = self.validate()

            self.log_epoch_results(epoch, train_loss, train_metrics_res, val_loss, val_metrics)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint()

        # tmp
        self.close()

    def validate(self) -> tuple[float, dict[str, torch.Tensor]]:
        self.model.eval()
        self.validation_metrics.reset()
        val_loss = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss(output, target).item()
                self.validation_metrics.update(output, target)
        val_metrics = self.validation_metrics.compute()
        return val_loss / len(self.val_loader), val_metrics
    
    def save_checkpoint(self) -> None:
        os.makedirs(os.path.dirname(f"results/"), exist_ok=True)
        model_path = f"results/{self.model_path}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    def log_epoch_results(self, epoch: int, train_loss: float, train_metrics: dict[str, torch.Tensor], val_loss: float, val_metrics: dict[str, torch.Tensor]) -> None:
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f"Metrics/Train_{metric_name}", value, epoch)
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f"Metrics/Validation_{metric_name}", value, epoch)
            
        print(f"Epoch {epoch + 1}/{self.args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}") 

    def close(self) -> None:
        self.writer.close()

class Evaluator:
    """Handles testing of the model."""

    def __init__(self, args: argparse.Namespace, model: nn.Module, test_loader: DataLoader) -> None:
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.loss = nn.MSELoss()
        self.test_metrics = METRICS.clone().to(self.device)
        
        if args.model != 'FNO':
            self.model_path = f"{args.model}_{args.upscale_factor}_{args.crop_size}_{args.epochs}_{args.lr}"
        else:
            self.model_path = f"{args.model}_{args.upscale_factor}_{args.crop_size}_{args.epochs}_{args.lr}_{args.modes}_{args.width}_{args.layers}"
        
        self.load_model()
       
        log_dir = os.path.join(args.logdir, self.model_path) 
        self.writer = SummaryWriter(log_dir=log_dir)

    def load_model(self) -> None:
        checkpoint = torch.load(f"results/{self.model_path}.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print('Model loaded...')

    def test(self) -> None:
        print("Starting evaluation...")
        self.model.eval()
        self.test_metrics.reset()
        test_loss = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).items()
                self.test_metrics.update(output, target)
    
        test_loss /= len(self.test_loader)
        test_metrics = self.test_metrics.compute()
        self.log_testing_results(test_loss, test_metrics)

        #tmp
        self.close()

    def log_testing_results(self, test_loss: float, test_metrics: dict[str, torch.Tensor]) -> None:
        for metric_name, value in test_metrics.items():
            self.writer.add_scalar(f"Metrics/Test_{metric_name}", value)
        print(f"Test MSE loss: {test_loss}, metrics: {test_metrics}")

    def close(self) -> None:
        self.writer.close()

