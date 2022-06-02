from torchvision.datasets import ImageNet
from torchvision import transforms

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class ImageNetWrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config)
        train_transforms = None
        if data_config.augment:
            train_transforms = self.get_train_transforms()
        self.ds_train = ImageNet(self.root, split='train',
                                transform=train_transforms)
        self.ds_val = None
        self.ds_test = ImageNet(self.root, split='val')
    

