import os
import torch
import numpy as np
from torchvision import datasets, transforms

from typing import Iterable

from biasedbed.datasets.abstract_dataloader import AbstractDataLoader


class DataLoader(AbstractDataLoader):
    num_classes = 16
    num_channels = 3

    def __init__(self, rank: int, config, train: bool):
        super(DataLoader, self).__init__(rank=rank,
                                         config=config,
                                         dataset_cfg_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
                                         train=train)
        self.dataset, self.sampler = self._init_dataset()

    def train_loader(self) -> Iterable:
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.config.training.batch_size,
                                                  num_workers=self.config.num_workers,
                                                  sampler=sampler)
        return data_loader

    def train_epoch_start(self, epoch: int):
        if self.config.distributed:
            self.sampler.set_epoch(epoch)
        else:
            pass

    def val_loader(self) -> Iterable:
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.config.training.batch_size,
                                                  num_workers=self.config.num_workers,
                                                  shuffle=False)
        return data_loader

    def test_loader(self) -> Iterable:
        return self.val_loader()

    def _init_dataset(self):
        path = f'{self.config.data_dir}/Silhouette/'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]),
                                 std=np.array([0.229, 0.224, 0.225]))
        ])
        dataset = datasets.ImageFolder(root=os.path.join(path, 'data'), transform=transform)

        if self.config.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.rank,
                drop_last=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
        return dataset, sampler
