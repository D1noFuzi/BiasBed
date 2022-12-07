import os
import importlib

from types import ModuleType


class Datasets:

    def __init__(self):
        self.datasets = dict()
        self._import_all()

    def __call__(self, dataset: str):
        if dataset not in self.datasets.keys():
            raise NotImplementedError("Dataset {} not available.".format(dataset))
        return self.datasets[dataset]

    def register(self, mod: ModuleType):
        dataset = mod.__name__.split(".")[2]
        if dataset in self.datasets.keys():
            raise ImportError("Dataset {} already exists. Please rename dataset directory.".format(dataset))
        self.datasets[dataset] = mod.DataLoader

    def _import_all(self):
        datasets = [f.name for f in os.scandir(os.path.dirname(__file__)) if ((f.is_dir()) & ("__" not in str(f)))]
        for dataset in datasets:
            self._import(dataset)

    def _import(self, dataset: str):
        if dataset in self.datasets.keys():
            return
        try:
            mod = importlib.import_module(f'biasedbed.datasets.{dataset}.dataloader')
            self.register(mod)
        except ImportError as e:
            print('Skipping dataset {} due to '.format(dataset))
            print(e)
            # raise


