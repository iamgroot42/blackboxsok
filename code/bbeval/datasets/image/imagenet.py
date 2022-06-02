from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.config import DatasetConfig


class ImageNetWrapper(CustomDatasetWrapper):
    def __init__(self, data_config: DatasetConfig):
        super().__init__(data_config)
        self.train_transforms = transforms.Compose([
            # Maybe consider making these sizes configurable
            transforms.Resize(299),
            transforms.RandomCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        self.test_transforms = self.train_transforms
        if data_config.augment:
            self.train_transforms = self.get_train_transforms()
        # self.ds_train = ImageNet(self.root, split='train',
        #                         transform=train_transforms)
        self.ds_test = ImageNet(self.root, split='val',
                                transform=self.test_transforms)

