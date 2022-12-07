import torch
import wandb

from omegaconf import OmegaConf


class AbstractAlgorithm(torch.nn.Module):

    def __init__(self, rank: int, num_classes: int, num_channels: int, config: OmegaConf, algo_cfg_path: str):
        super(AbstractAlgorithm, self).__init__()
        self.rank = rank
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.config = config
        self.algorithm_cfg = self._load_config(algo_cfg_path)
        self.model = self._init_model()

    def update(self, x, y):
        """
        Single batch update of network. You need to compute losses here, call backward and update metrics
        :param x: input e.g. NxCxWxH for images
        :param y: target labels, e.g Nx1
        :return: None
        """
        raise NotImplementedError

    def train_epoch_end(self):
        """
        Anything you want to do at the end of training one epoch, e.g. scheduler step etc.
        Can be ignored if not needed.
        :return: None
        """
        pass

    def val(self, x, y):
        """
        Validation loop over an entry epoch.
        :param x: input e.g. NxCxWxH for images
        :param y: target labels, e.g Nx1
        :return: None
        """
        raise NotImplementedError

    def test(self, x, y, dataset: str, mapping):
        """
        Test loop over an entry epoch.
        :param x: input e.g. NxCxWxH for images
        :param y: target labels, e.g Nx1
        :param dataset: dataset name for logging
        :param mapping: mapping function to project output classes to target classes
        :return: None
        """
        raise NotImplementedError

    def log_train(self, step: int):
        """
        Log training metrics at the end of each training epoch.
        :param step: epoch
        :return: None
        """
        pass

    def log_val(self, step: int):
        """
        Log validation metrics at the end of each validation epoch.
        :param step: epoch
        :return: None
        """
        pass

    def log_test(self, step: int):
        """
        Log test metrics at the end of each test epoch.
        :param step: epoch
        :return: None
        """
        pass

    def save_model(self, step: int, path: str):
        """
        Save your model, e.g. model.state_dict(), optimizer.state_dict() etc.
        :param step: epoch
        :param path: save path
        :return:
        """
        raise NotImplementedError

    def _init_model(self):
        """
        Initialize your model
        :return: your model
        """
        raise NotImplementedError

    def _init_metrics(self):
        """
        Initialize metrics you want to use during training, validation and testing
        :return: your metrics
        """
        pass

    def _load_config(self, algo_cfg_path):
        cfg = OmegaConf.load(algo_cfg_path)
        wcfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wcfg = {'algorithm': {**OmegaConf.to_container(self.config.algorithm, resolve=True, throw_on_missing=True), **wcfg}}
        if self.rank == 0:
            wandb.config.update(wcfg, allow_val_change=True)
        return cfg
