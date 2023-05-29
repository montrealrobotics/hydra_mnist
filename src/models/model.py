from typing import Dict, List, Optional, Tuple, Union
import functools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

import pytorch_lightning as pl

from src.utils import get_logger

log = get_logger(__name__)


class MNISTModel(nn.Module):
    def __init__(self, hidden_size: int = 64, p_dropout: float = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


class LitMNIST(pl.LightningModule):
    def __init__(self, net, learning_rate=2e-4):
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Set our init args as class attributes
        self.learning_rate = learning_rate

        # Model
        self.net = net

        self.val_accuracy = MulticlassAccuracy(num_classes=10)
        self.test_accuracy = MulticlassAccuracy(num_classes=10)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

