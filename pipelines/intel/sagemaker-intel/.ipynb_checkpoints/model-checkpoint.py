

from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Dict, Optional, Tuple, List

import argparse
from torchmetrics import F1Score, Precision, Recall, ConfusionMatrix, MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime


weight_decay=5e-4


class LitResnet(pl.LightningModule):

    def __init__(
            self,
            model_name='resnet18',
            optim_name="SGD",
            num_classes=6,
            lr=0.001,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.num_classes = num_classes

        self.net = timm.create_model(model_name, pretrained=True, num_classes=self.num_classes)
        self.lr = lr
        # configure optimizer
        if optim_name == "ADAM":
            self.optim_name = torch.optim.Adam
        if optim_name == "SGD":
            self.optim_name = torch.optim.SGD
        if optim_name == "RMS":
            self.optim_name = torch.optim.RMSprop

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=self.num_classes)

        # some other metrics to be logged
        self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.precision_score = Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.recall_score = Recall(task="multiclass", average='macro', num_classes=self.num_classes)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.f1_score(preds, targets)
        self.precision_score(preds, targets)
        self.recall_score(preds, targets)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/f1", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/precision", self.precision_score, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/recall", self.recall_score, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = self.optim_name(
            self.parameters(),
            lr=self.lr,
        )
#         steps_per_epoch = 45000 // BATCHSIZE
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size  = 10 , gamma = 0.5)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train/loss",
                
            }
          }
    