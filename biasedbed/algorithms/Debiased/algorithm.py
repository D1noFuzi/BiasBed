import os

import wandb

from biasedbed.algorithms.abstract_algorithm import AbstractAlgorithm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.sgd import SGD
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from torchvision.models import resnet50

from torchmetrics import Accuracy, MeanMetric

from .utils.AdaIN import StyleTransfer
from .utils.aux_bn import MixBatchNorm2d, to_mix_status, to_clean_status


class Algorithm(AbstractAlgorithm):

    def __init__(self, rank: int, num_classes: int, num_channels: int, config):
        super(Algorithm, self).__init__(rank=rank,
                                        num_classes=num_classes,
                                        num_channels=num_channels,
                                        config=config,
                                        algo_cfg_path=os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.scaler = GradScaler(enabled=self.config.distributed)
        self.styletransfer = self._init_style_transfer()
        self.optimizer, self.scheduler = self._init_optimizer()
        self.train_metrics, self.val_metrics, self.test_metrics = self._init_metrics()
        self.loss = CrossEntropyLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.rank) * 255.
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.rank) * 255.

    def update(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)

        # Set MixBatchNorm to mix status
        self.model.apply(to_mix_status)

        # Undo normalization for style transfer
        # x = x * self.std[:, None, None] + self.mean[:, None, None]
        with autocast(enabled=self.algorithm_cfg.mixedprec):
            inputs, targets = self.styletransfer(x,
                                                 y,
                                                 alpha=0.5,
                                                 label_mix_alpha=1. - self.algorithm_cfg.label_alpha,
                                                 replace=False)
            # inputs = (inputs - self.mean[:, None, None]) / self.std[:, None, None]
            if not torch.isfinite(inputs).all():
                # pbar.write('Found invalid value in input tensor. Skipping this batch.')
                print('Found invalid value in input tensor. Skipping this batch.')
                return
            outputs = self.model(inputs)

            loss = (1. - targets[2]) * self.loss(outputs, targets[0]) + targets[2] * self.loss(outputs, targets[1])
            loss = loss.mean()
            targets = targets[0]

        # Backward loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Compute metrics
        preds = torch.argmax(outputs, dim=1)
        self.train_metrics['acc'](preds, targets)
        self.train_metrics['loss'](loss)

    def train_epoch_end(self):
        self.scheduler.step()

    def val(self, x, y):
        self.model.apply(to_clean_status)
        with torch.no_grad():
            with autocast(enabled=self.algorithm_cfg.mixedprec):
                # TODO: implement your validation routine here
                logits = self.model(x)
                loss = self.loss(logits, y)
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics['acc'](preds, y)
        self.val_metrics['loss'](loss)

    def test(self, x, y, dataset: str, mapping):
        self.model.apply(to_clean_status)
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
        optimizer = SGD(self.model.parameters(),
                        lr=self.algorithm_cfg.learning_rate,
                        momentum=self.algorithm_cfg.momentum,
                        weight_decay=self.algorithm_cfg.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=self.algorithm_cfg.milestones, gamma=self.algorithm_cfg.gamma)
        return optimizer, scheduler

    def _init_model(self):
        if self.algorithm_cfg.backbone == 'resnet50':
            model = resnet50(pretrained=False, num_classes=self.num_classes, norm_layer=MixBatchNorm2d)
        else:
            raise NotImplementedError(f'Backbone {self.algorithm_cfg.backbone} not implemented.')
        return model

    def _init_style_transfer(self):
        styletransfer = StyleTransfer(rank=self.rank,
                                      decoder_ckpt=os.path.join(os.path.dirname(__file__), './utils/decoder.pth'),
                                      vgg_ckpt=os.path.join(os.path.dirname(__file__), './utils/vgg_normalised.pth'))
        return styletransfer

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


