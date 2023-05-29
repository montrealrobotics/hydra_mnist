from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import glob
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils import get_logger

log = get_logger(__name__)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 8, drop_last: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)
