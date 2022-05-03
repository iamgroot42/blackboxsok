from torchvision.datasets import MNIST

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class MNISTWrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config)
        self.ds_train = MNIST(self.root, classes='train')
        self.ds_val = MNIST(self.root, classes='val')
        self.ds_test = MNIST(self.root, classes='test')
