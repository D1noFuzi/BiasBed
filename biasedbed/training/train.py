import torch
import wandb

from pathlib import Path

from omegaconf import OmegaConf

from biasedbed.training.trainer import Trainer
from biasedbed.algorithms.algorithms import Algorithms
from biasedbed.datasets.datasets import Datasets


class Training:

    def __init__(self, config):
        self.config = config
        self.algorithm = self._get_algorithm_class()

    def launch_training(self):
        if self.config.distributed:
            torch.multiprocessing.spawn(self.run, nprocs=self.config.world_size, join=True)
        else:
            self.run(gpu=0)

    def run(self, gpu):
        if gpu == 0:
            wcfg = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
            log_path = Path(self.config.logging.wandb_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            wandb.init(project=self.config.project,
                       entity='biasedbed',
                       dir=str(log_path),
                       config=wcfg,
                       resume='allow',
                       mode=self.config.logging.wandb_mode,
                       settings=wandb.Settings(start_method="fork"))
        train_dataset, test_datasets = self._get_dataset_classes(gpu=gpu)
        trainer = Trainer(rank=gpu,
                          config=self.config,
                          model=self.algorithm,
                          train_dataset=train_dataset,
                          test_datasets=test_datasets)
        trainer.train()

        if self.config.distributed:
            trainer.cleanup_distributed()
        if gpu == 0:
            wandb.finish()

    def _get_algorithm_class(self):
        return Algorithms()(self.config.algorithm.name)

    def _get_dataset_classes(self, gpu):
        dataset_coll = Datasets()
        train_dataset = dataset_coll(self.config.training.dataset.name)(rank=gpu, config=self.config, train=True)
        test_datasets = dict()
        for dataset in self.config.testing.datasets:
            test_datasets[dataset] = dataset_coll(dataset)(rank=gpu, config=self.config, train=False)
        return train_dataset, test_datasets
