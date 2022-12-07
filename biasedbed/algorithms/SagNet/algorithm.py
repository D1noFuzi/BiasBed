"""
Adapted from https://github.com/hyeonseobnam/sagnet/
"""

import os

import wandb

from biasedbed.algorithms.abstract_algorithm import AbstractAlgorithm

import torch
from torch import optim
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from torchmetrics import Accuracy, MeanMetric


os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '..', './pretrained_models/')


class Algorithm(AbstractAlgorithm):

    def __init__(self, rank: int, num_classes: int, num_channels: int, config):
        super(Algorithm, self).__init__(rank=rank,
                                        num_classes=num_classes,
                                        num_channels=num_channels,
                                        config=config,
                                        algo_cfg_path=os.path.join(os.path.dirname(__file__), 'config.yaml'))
        self.scaler = GradScaler(enabled=self.config.distributed)
        self.optimizer, self.optimizer_style, self.optimizer_adv, self.scheduler, self.scheduler_style, self.scheduler_adv = self._init_optimizer()
        self.train_metrics, self.val_metrics, self.test_metrics = self._init_metrics()
        self.loss = CrossEntropyLoss()
        self.criterion_adv = AdvLoss()

    def update(self, x, y):

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_style.zero_grad()
        self.optimizer_adv.zero_grad()

        with autocast(enabled=self.algorithm_cfg.mixedprec):
            logits, logits_style = self.model(x)
        # learn style
        loss_style = self.loss(logits_style, y)
        self.scaler.scale(loss_style).backward(retain_graph=True)

        # learn style_adv
        loss_adv = self.algorithm_cfg.w_adv * self.criterion_adv(logits_style)
        self.scaler.scale(loss_adv).backward(retain_graph=True)
        if self.algorithm_cfg.clip_adv is not None:
            if self.config.distributed:
                torch.nn.utils.clip_grad_norm_(self.model.module.adv_params(), self.algorithm_cfg.clip_adv)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.adv_params(), self.algorithm_cfg.clip_adv)

        # learn content
        loss = self.loss(logits, y)
        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer_style)
        self.scaler.step(self.optimizer_adv)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        #
        # # SagNet uses iterations not epoch milestones
        # self.scheduler.step()
        # self.scheduler_style.step()
        # self.scheduler_adv.step()

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_metrics['acc'](preds, y)
        self.train_metrics['loss'](loss)
        self.train_metrics['loss_adv'](loss_adv)
        self.train_metrics['loss_style'](loss_style)

    def train_epoch_end(self):
        # SagNet with epoch milestones...
        self.scheduler.step()
        self.scheduler_style.step()
        self.scheduler_adv.step()

    def val(self, x, y):
        with torch.no_grad():
            with autocast(enabled=self.algorithm_cfg.mixedprec):
                # TODO: implement your validation routine here
                logits, _ = self.model(x)
                loss = self.loss(logits, y)
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_metrics['acc'](preds, y)
        self.val_metrics['loss'](loss)

    def test(self, x, y, dataset: str, mapping):
        # Compute metrics
        with torch.no_grad():
            with autocast(enabled=self.algorithm_cfg.mixedprec):
                logits, _ = self.model(x)
                preds = mapping(logits)
        self.test_metrics[f'{dataset}/acc'](preds, y)

    def log_train(self, step: int):
        acc = self.train_metrics['acc'].compute()
        loss = self.train_metrics['loss'].compute()
        loss_adv = self.train_metrics['loss_adv'].compute()
        loss_style = self.train_metrics['loss_style'].compute()
        if self.rank == 0:
            wandb.log({'train/acc': acc, 'train/loss': loss, 'train/loss_adv': loss_adv, 'train/loss_style': loss_style}, step=step)
        self.train_metrics['acc'].reset()
        self.train_metrics['loss'].reset()
        self.train_metrics['loss_adv'].reset()
        self.train_metrics['loss_style'].reset()

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
        optim_hyperparams = {'lr': self.algorithm_cfg.learning_rate,
                             'weight_decay': self.algorithm_cfg.weight_decay,
                             'momentum': self.algorithm_cfg.momentum}
        if self.algorithm_cfg.scheduler == 'step':
            Scheduler = optim.lr_scheduler.MultiStepLR
            sch_hyperparams = {'milestones': self.algorithm_cfg.milestones,
                               'gamma': self.algorithm_cfg.gamma}
        elif self.algorithm_cfg.scheduler == 'cosine':
            Scheduler = optim.lr_scheduler.CosineAnnealingLR
            sch_hyperparams = {'T_max': self.algorithm_cfg.iterations}

        # Main learning
        params = self.model.parameters()
        optimizer = optim.SGD(params, **optim_hyperparams)
        scheduler = Scheduler(optimizer, **sch_hyperparams)

        # Style learning
        params_style = self.model.style_params()
        optimizer_style = optim.SGD(params_style, **optim_hyperparams)
        scheduler_style = Scheduler(optimizer_style, **sch_hyperparams)

        # Adversarial learning
        params_adv = self.model.adv_params()
        optimizer_adv = optim.SGD(params_adv, **optim_hyperparams)
        scheduler_adv = Scheduler(optimizer_adv, **sch_hyperparams)

        return optimizer, optimizer_style, optimizer_adv, scheduler, scheduler_style, scheduler_adv

    def _init_model(self):
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }

        if self.algorithm_cfg.depth == 18:
            model = ResNet(BasicBlock, [2, 2, 2, 2],
                           self.num_classes,
                           self.algorithm_cfg.dropout,
                           self.algorithm_cfg.style_stage)
        elif self.algorithm_cfg.depth == 50:
            model = ResNet(Bottleneck, [3, 4, 6, 3],
                           self.num_classes,
                           self.algorithm_cfg.dropout,
                           self.algorithm_cfg.style_stage)
        elif self.algorithm_cfg.depth == 101:
            model = ResNet(Bottleneck, [3, 4, 23, 3],
                           self.num_classes,
                           self.algorithm_cfg.dropout,
                           self.algorithm_cfg.style_stage)
        elif self.algorithm_cfg.depth == 152:
            model = ResNet(Bottleneck, [3, 8, 36, 3],
                           self.num_classes,
                           self.algorithm_cfg.dropout,
                           self.algorithm_cfg.style_stage)

        if self.algorithm_cfg.pretrained:
            model_url = model_urls['resnet' + str(self.algorithm_cfg.depth)]
            print('load a pretrained model from {}'.format(model_url))

            states = model_zoo.load_url(model_url)
            states.pop('fc.weight')
            states.pop('fc.bias')
            model.load_state_dict(states, strict=False)

            if model.sagnet:
                states_style = {}
                for i in range(model.style_stage, 5):
                    for k, v in states.items():
                        if k.startswith('layer' + str(i)):
                            states_style[str(i - model.style_stage) + k[6:]] = v
                model.style_net.load_state_dict(states_style)

        return model

    def _init_metrics(self):
        train_metrics = {
            'acc': Accuracy().to(self.rank),
            'loss': MeanMetric().to(self.rank),
            'loss_adv': MeanMetric().to(self.rank),
            'loss_style': MeanMetric().to(self.rank)
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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, drop=0, sagnet=True, style_stage=3):
        super().__init__()

        self.drop = drop
        self.sagnet = sagnet
        self.style_stage = style_stage

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.drop)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.sagnet:
            # randomizations
            self.style_randomization = Randomizer(alpha=True)
            self.content_randomization = Randomizer(alpha=False)

            # style-biased network
            style_layers = []
            if style_stage == 1:
                self.inplanes = 64
                style_layers += [self._make_layer(block, 64, layers[0])]
            if style_stage <= 2:
                self.inplanes = 64 * block.expansion
                style_layers += [self._make_layer(block, 128, layers[1], stride=2)]
            if style_stage <= 3:
                self.inplanes = 128 * block.expansion
                style_layers += [self._make_layer(block, 256, layers[2], stride=2)]
            if style_stage <= 4:
                self.inplanes = 256 * block.expansion
                style_layers += [self._make_layer(block, 512, layers[3], stride=2)]
            self.style_net = nn.Sequential(*style_layers)

            self.style_avgpool = nn.AdaptiveAvgPool2d(1)
            self.style_dropout = nn.Dropout(self.drop)
            self.style_fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adv_params(self):
        params = []
        layers = [self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers[:self.style_stage]:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    params += [p for p in m.parameters()]
        return params

    def style_params(self):
        params = []
        for m in [self.style_net, self.style_fc]:
            params += [p for p in m.parameters()]
        return params

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, stride=stride,
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            if self.sagnet and i + 1 == self.style_stage and self.training:
                # randomization
                x_style = self.content_randomization(x)
                x = self.style_randomization(x)
            x = layer(x)

        # content output
        feat = self.avgpool(x)
        feat = feat.view(x.size(0), -1)
        feat = self.dropout(feat)
        y = self.fc(feat)

        if self.sagnet and self.training:
            # style output
            x_style = self.style_net(x_style)
            feat = self.style_avgpool(x_style)
            feat = feat.view(feat.size(0), -1)
            feat = self.style_dropout(feat)
            y_style = self.style_fc(feat)
        else:
            y_style = None

        return y, y_style


class Randomizer(nn.Module):
    def __init__(self, alpha=False, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            if self.alpha:
                alpha = torch.rand(N, 1, 1)
            else:
                alpha = torch.ones(1)
            if x.is_cuda:
                alpha = alpha.cuda()
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class AdvLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, inputs):
        inputs = inputs.softmax(dim=1)
        loss = - torch.log(inputs + self.eps).mean(dim=1)
        return loss.mean()