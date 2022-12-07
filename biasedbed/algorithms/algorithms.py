import os
import importlib

from types import ModuleType


class Algorithms:

    def __init__(self):
        self.algorithms = dict()
        self._import_all()

    def __call__(self, algorithm: str):
        if algorithm not in self.algorithms.keys():
            raise NotImplementedError("Algorithm {} not implemented.".format(algorithm))
        return self.algorithms[algorithm]

    def register(self, mod: ModuleType):
        algorithm = mod.__name__.split(".")[2]
        if algorithm in self.algorithms.keys():
            raise ImportError("Algorithm {} already exists. Please rename algorithm directory.".format(algorithm))
        self.algorithms[algorithm] = mod.Algorithm

    def _import_all(self):
        algorithms_names = [f.name for f in os.scandir(os.path.dirname(__file__)) if
                            ((f.is_dir()) & ("__" not in str(f)) & ("abstract_algorithm" not in str(f)))]
        for algorithm in algorithms_names:
            self._import(algorithm)

    def _import(self, algorithm: str):
        if algorithm in self.algorithms.keys():
            return
        try:
            mod = importlib.import_module(f'biasedbed.algorithms.{algorithm}.algorithm')
            self.register(mod)
        except ImportError as e:
            print('Skipping algorithm {} due to '.format(algorithm))
            print(e)


