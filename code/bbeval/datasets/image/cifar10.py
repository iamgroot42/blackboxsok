from torchvision.datasets import CIFAR10

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class CIFAR10Wrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config)
        train_transforms = None
        if data_config.augment:
            train_transforms = self.get_train_transforms()
        self.ds_train = CIFAR10(self.root, train=True,
                                transform=train_transforms,
                                download=True)
        self.ds_val = None
        self.ds_test = CIFAR10(self.root, train=False, download=True)

