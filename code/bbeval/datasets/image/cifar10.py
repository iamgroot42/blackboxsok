from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class CIFAR10Wrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config)
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.test_transforms = self.train_transforms
        if data_config.augment:
            self.train_transforms = self.get_train_transforms()
        self.ds_train = CIFAR10(self.root, train=True,
                                transform=self.train_transforms,
                                download=True)
        self.ds_test = CIFAR10(self.root, train=False,
                               transform=self.test_transforms,
                               download=True)
        self.ds_val = None
