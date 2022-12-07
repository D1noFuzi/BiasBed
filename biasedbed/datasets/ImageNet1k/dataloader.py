import torch
import os

import numpy as np

from pathlib import Path
from typing import List, Iterable

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from biasedbed.datasets.abstract_dataloader import AbstractDataLoader


class DataLoader(AbstractDataLoader):

    num_classes = 1000
    num_channels = 3

    def __init__(self, rank: int, config, train: bool):
        super(DataLoader, self).__init__(rank=rank,
                                         config=config,
                                         dataset_cfg_path=os.path.join(os.path.dirname(__file__), 'config.yaml'),
                                         train=train)

        self.training_path = Path(config.data_dir) / './ImageNet1k/data/dataset_train_jpg_100_256.beton'
        self.validation_path = Path(config.data_dir) / './ImageNet1k/data/dataset_val_jpg_100_256.beton'
        # self.testing_path = Path('')  # not needed for imagenet1k

        self.imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
        self.imagenet_std = np.array([0.229, 0.224, 0.225]) * 255
        self.default_crop_ratio = 224/256
        self.res = 256

    def train_loader(self) -> Iterable:
        train_path = Path(self.training_path)
        assert train_path.is_file()

        decoder = RandomResizedCropRGBImageDecoder((self.res, self.res),
                                                   scale=(0.5, 1.0), ratio=(0.8, 1.0))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(self.rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.imagenet_mean, self.imagenet_std, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.rank), non_blocking=True)
        ]

        order = OrderOption.RANDOM if self.config.distributed else OrderOption.QUASI_RANDOM

        tinfo = np.iinfo('int32')
        seed = np.random.randint(0, tinfo.max)

        loader = Loader(self.training_path,
                        batch_size=self.config.training.batch_size,
                        num_workers=self.config.num_workers,
                        order=order,
                        os_cache=self.dataset_cfg.in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=self.config.distributed,
                        seed=seed)

        return loader

    def val_loader(self) -> Iterable:
        val_path = Path(self.validation_path)
        assert val_path.is_file()

        res_tuple = (self.res, self.res)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=self.default_crop_ratio)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(self.rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.imagenet_mean, self.imagenet_std, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.rank), non_blocking=True)
        ]

        loader = Loader(self.validation_path,
                        batch_size=self.config.validation.batch_size,
                        num_workers=self.config.num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=True,
                        os_cache=self.dataset_cfg.in_memory,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=self.config.distributed)
        return loader

    def test_loader(self) -> Iterable:
        # We return imagenet validation here
        return self.val_loader()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
