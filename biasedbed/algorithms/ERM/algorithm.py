import os

import wandb

from biasedbed.algorithms.abstract_algorithm import AbstractAlgorithm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from torchvision.models import resnet50

from torchmetrics import Accuracy, MeanMetric


class Algorithm(AbstractAlgorithm):

    def __init__(self, rank: int, num_classes: int, num_channels: int, config):
        super(Algorithm, self).__init__(rank=rank,
                                        num_classes=num_classes,
                                        num_channels=num_channels,
                                        config=config,
                                        algo_cfg_path=os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.scaler = GradScaler(enabled=self.config.distributed)
        self.optimizer, self.scheduler = self._init_optimizer()
        self.train_metrics, self.val_metrics, self.test_metrics = self._init_metrics()
        self.loss = CrossEntropyLoss()

    def update(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.algorithm_cfg.mixedprec):
            logits = self.model(x)
            loss = self.loss(logits, y)

        # Backward loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_metrics['acc'](preds, y)
        self.train_metrics['loss'](loss)

    def train_epoch_end(self):
        self.scheduler.step()

    def val(self, x, y):
        with torch.no_grad():
            with autocast(enabled=self.algorithm_cfg.mixedprec):
                logits = self.model(x)
                loss = self.loss(logits, y)
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics['acc'](preds, y)
        self.val_metrics['loss'](loss)

    def test(self, x, y, dataset: str, mapping):
        # Compute metrics
        with torch.no_grad():
            with autocast(enabled=self.algorithm_cfg.mixedprec):
                logits = self.model(x)
                preds = mapping(logits)
        self.test_metrics[f'{dataset}/acc'](preds, y)

    def log_train(self, step: int):
        acc = self.train_metrics['acc'].compute()
        loss = self.train_metrics['loss'].compute()
        if self.rank == 0:
            wandb.log({'train/acc': acc, 'train/loss': loss}, step=step)
        self.train_metrics['acc'].reset()
        self.train_metrics['loss'].reset()

    def log_val(self, step: int):
        acc = self.val_metrics['acc'].compute()
        loss = self.val_metrics['loss'].compute()
        if self.rank == 0:
            wandb.log({'val/acc': acc, 'val/loss': loss}, step=step)
        self.val_metrics['acc'].reset()
        self.val_metrics['loss'].reset()

    def log_test(self, step: int):
        for dataset in self.config.testing.datasets:
            acc = self.test_metrics[f'{dataset}/acc'].compute()
            if self.rank == 0:
                wandb.log({f'test/{dataset}/acc': acc}, step=step)
            self.test_metrics[f'{dataset}/acc'].reset()

    def save_model(self, step: int, path: str):
        torch.save({
            'epoch': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(path, f'epoch_{step}.pth'))

    def _init_optimizer(self):
        if self.algorithm_cfg.optimizer == 'Adam':
            optimizer = Adam(params=self.model.parameters(),
                             lr=self.algorithm_cfg.learning_rate,
                             weight_decay=self.algorithm_cfg.weight_decay)
        elif self.algorithm_cfg.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(),
                            lr=self.algorithm_cfg.learning_rate,
                            momentum=self.algorithm_cfg.momentum,
                            weight_decay=self.algorithm_cfg.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {self.algorithm_cfg.optimizer} not implemented.')
        scheduler = MultiStepLR(optimizer, milestones=self.algorithm_cfg.milestones, gamma=self.algorithm_cfg.gamma)
        return optimizer, scheduler

    def _init_model(self):
        if self.algorithm_cfg.backbone == 'resnet50':
            return resnet50(pretrained=False, num_classes=self.num_classes)
        elif self.algorithm_cfg.backbone == 'Net':
            return Net(self.num_channels, self.num_classes)
        else:
            raise NotImplementedError(f'Backbone {self.algorithm_cfg.backbone} not implemented.')

    def _init_metrics(self):
        train_metrics = {
            'acc': Accuracy().to(self.rank),
            'loss': MeanMetric().to(self.rank)
        }
        if self.rank == 0:
            wandb.define_metric("train/acc", summary="max")
        val_metrics = {
            'acc': Accuracy().to(self.rank),
            'loss': MeanMetric().to(self.rank)
        }
        if self.rank == 0:
            wandb.define_metric("val/acc", summary="max")
        test_metrics = {}
        for dataset in self.config.testing.datasets:
            test_metrics[f'{dataset}/acc'] = Accuracy().to(self.rank)
            if self.rank == 0:
                wandb.define_metric(f"test/{dataset}/acc", summary="max")
        return train_metrics, val_metrics, test_metrics


class Net(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output