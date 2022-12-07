import os
import wandb
from omegaconf import OmegaConf, DictConfig, ListConfig

from typing import Union


class Sweep:

    def __init__(self, config: Union[DictConfig, ListConfig]):
        self.config = config
        self.sweep_cfg = self._init_sweep_config()

    def _init_sweep_config(self):
        algorithm_sweep_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'algorithms',
                                                          self.config.algorithm.name, 'sweep.yaml'))
        sweep_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'sweep.yaml'))
        sweep_cfg.merge_with(algorithm_sweep_cfg)
        return OmegaConf.to_container(sweep_cfg)

    def init_sweep(self) -> str:
        return wandb.sweep(self.sweep_cfg, project=self.config.project)