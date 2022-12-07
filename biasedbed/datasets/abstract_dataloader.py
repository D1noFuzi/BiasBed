import os
import torch
import wandb
from pathlib import Path
from torchvision import datasets, transforms

from typing import Iterable, Optional, Union
from omegaconf import OmegaConf, DictConfig, ListConfig


class AbstractDataLoader:

    def __init__(self, rank: int, config, dataset_cfg_path: str, train: bool):
        self.rank = rank
        self.train = train
        self.config = config
        self.dataset_cfg = self._load_config(dataset_cfg_path)

    def train_loader(self) -> Iterable:
        raise NotImplementedError

    def val_loader(self) -> Iterable:
        raise NotImplementedError

    def test_loader(self) -> Iterable:
        raise NotImplementedError

    def train_epoch_start(self):
        pass

    @property
    def num_classes(self):
        raise NotImplementedError
    
    @property
    def num_channels(self):
        raise NotImplementedError

    def _load_config(self, dataset_cfg_path) -> Optional[Union[DictConfig, ListConfig]]:
        if Path(dataset_cfg_path).is_file():
            cfg = OmegaConf.load(dataset_cfg_path)
            if self.train and self.rank == 0:
                wcfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                wcfg = {'training': {**OmegaConf.to_container(self.config.training, resolve=True, throw_on_missing=True), **wcfg}}
                wandb.config.update(wcfg, allow_val_change=True)
        else:
            cfg = None
        return cfg
