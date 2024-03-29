from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class MNISTWrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config, num_classes=10)
        self.train_transforms = transforms.Compose([
            transforms.ToTensor()])
        self.test_transforms = self.train_transforms
        if data_config.augment:
            self.train_transforms = self.get_train_transforms()
        self.ds_train = MNIST(self.root, train=True,
                              transform=self.train_transforms, download=True)
        self.ds_test = MNIST(self.root, train=False,
                             transform=self.test_transforms, download=True)
