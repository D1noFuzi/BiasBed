# TODO: change DataLoader typing to template DataLoader!
import sys
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from biasedbed.algorithms.abstract_algorithm import AbstractAlgorithm
from biasedbed.datasets.MNIST.dataloader import DataLoader
from biasedbed.mappings.mappings import mappings as mps

from typing import Tuple, Iterable, Dict


class Trainer:

    def __init__(self,
                 rank: int,
                 config,
                 model: AbstractAlgorithm,
                 train_dataset: DataLoader,
                 test_datasets: Dict[str, DataLoader]) -> None:
        self.rank = rank
        self.config = config
        if self.config.distributed:
            self.setup_distributed(self.config.address, self.config.port, self.config.world_size)
        self.train_dataset = train_dataset
        self.test_datasets = test_datasets
        self.model = self.init_model(model)
        self.train_loader, self.val_loader, self.test_loaders, self.mappings = self.setup_datasets()
        if self.rank == 0:
            self.save_path = Path(self.config.training.save_dir) / f'run-{wandb.run.id}'
            self.save_path.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        for epoch in tqdm(range(self.config.training.epochs),
                          disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                          position=0,
                          file=sys.stderr,
                          leave=True,
                          desc='Epochs'):
            self.train_loop(epoch)
            if epoch % self.config.validation.interval == 0:
                self.val_loop(epoch)
            if epoch % self.config.testing.interval == 0:
                self.test_loop(epoch)
            if epoch % self.config.training.save_model_interval == 0 and self.rank == 0:
                self.model.save_model(epoch, str(self.save_path))
            sys.stderr.flush()

    def train_loop(self, epoch: int):
        self.model.train()
        # counter = 0
        self.train_dataset.train_epoch_start()  # E.g. set_epoch for distributed training
        iterator = iter(self.train_loader)
        for idx, (x, y) in tqdm(enumerate(iterator),
                                disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                                total=len(self.train_loader),
                                file=sys.stderr,
                                position=1,
                                leave=False,
                                desc='Train loop'):
            self.model.update(x.to(self.rank), y.to(self.rank))
            sys.stderr.flush()
            # counter += 1
            # if counter > 100:
            #     # iterator.close()
            #     break
        self.model.train_epoch_end()
        self.model.log_train(step=epoch)

    def val_loop(self, epoch: int):
        self.model.eval()
        iterator = iter(self.val_loader)
        for idx, (x, y) in tqdm(enumerate(iterator),
                                disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                                total=len(self.val_loader),
                                file=sys.stderr,
                                position=1,
                                leave=False,
                                desc='Val loop'):
            self.model.val(x.to(self.rank), y.to(self.rank))
            sys.stderr.flush()
        self.model.log_val(step=epoch)

    def test_loop(self, epoch: int):
        self.model.eval()
        for dataset, test_loader in self.test_loaders.items():
            iterator = iter(test_loader)
            for idx, (x, y) in tqdm(enumerate(iterator),
                                    disable=not (self.rank == 0 and self.config.logging.enable_tqdm),
                                    total=len(test_loader),
                                    file=sys.stderr,
                                    position=1,
                                    leave=False,
                                    desc=f'Test loop {dataset}'):
                self.model.test(x.to(self.rank), y.to(self.rank), dataset=dataset, mapping=self.mappings[(self.config.training.dataset.name, dataset)])
                sys.stderr.flush()
        self.model.log_test(step=epoch)

    def setup_datasets(self) -> Tuple[Iterable, Iterable, Dict[str, Iterable], Dict]:
        train_loader = self.train_dataset.train_loader()
        val_loader = self.train_dataset.val_loader()
        test_loaders = dict()
        mappings = dict()
        for test_dataset, test_dataset_cls in self.test_datasets.items():
            test_loaders[test_dataset] = test_dataset_cls.test_loader()
            if test_dataset_cls.num_classes != self.train_dataset.num_classes:
                try:
                    mappings[(self.config.training.dataset.name, test_dataset)] = mps[(self.config.training.dataset.name, test_dataset)]
                except ValueError as e:
                    raise e
            else:
                mappings[(self.config.training.dataset.name, test_dataset)] = mps[('Identity', 'Identity')]

        return train_loader, val_loader, test_loaders, mappings

    def init_model(self, model: AbstractAlgorithm) -> AbstractAlgorithm:
        model = model(rank=self.rank,
                      num_classes=self.train_dataset.num_classes,
                      num_channels=self.train_dataset.num_channels,
                      config=self.config)
        model = model.to(self.rank)
        if self.config.distributed:
            model.model = DDP(model.model,
                              device_ids=[self.rank],
                              find_unused_parameters=self.config.find_unused_parameters)
        return model

    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.rank, world_size=world_size)
        torch.cuda.set_device(self.rank)

    @staticmethod
    def cleanup_distributed():
        dist.destroy_process_group()
