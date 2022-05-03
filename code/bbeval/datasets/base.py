import os
import numpy as np
from torch.utils.data import DataLoader

from bbeval.config import DatasetConfig, TrainConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.datasets.utils import worker_init_fn


class CustomDatasetWrapper:
    def __init__(self, data_config: DatasetConfig):
        """
            self.ds_train and self.ds_test should be set to
            datasets to be used to train and evaluate.
        """
        self.classify = data_config.classify
        self.augment = data_config.augment
        self.root = data_config.root
    
    def get_train_transforms(self):
        return None

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: float = 1,
                    num_workers: int = 0,
                    prefetch_factor: int = 2):
        # This function should return new loaders at every call
        train_loader = DataLoader(
            self.ds_train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor
        )

        # Not all datasets will have val loaders
        if self.ds_val is None:
            val_loader = None
        else:
            val_loader = DataLoader(
                self.ds_val,
                batch_size=batch_size * val_factor,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                prefetch_factor=prefetch_factor
            )

        test_loader = DataLoader(
            self.ds_test,
            batch_size=batch_size * val_factor,
            shuffle=eval_shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor
        )

        return train_loader, val_loader, test_loader

    def get_save_dir(self, train_config: TrainConfig) -> str:
        """
            Return path to directory where models will be saved,
            for a given configuration.
        """
        raise NotImplementedError(
            "Function to fetch model save path not implemented")

    def get_save_path(self, train_config: TrainConfig, name: str) -> str:
        """
            Function to get prefix + name for saving
            the model.
        """
        prefix = self.get_save_dir(train_config)
        if name is None:
            return prefix
        return os.path.join(prefix, name)

    def load_model(self, path: str, on_cpu: bool = False) -> GenericModelWrapper:
        """Load model from a given path"""
        raise NotImplementedError("Function to load model not implemented")

    def __str__(self):
        return f"{type(self).__name__}(classify={self.classify})"
