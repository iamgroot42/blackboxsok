from bbeval.datasets.image import mnist, cifar10, imagenet
from bbeval.datasets.malware import ember
from bbeval.config import DatasetConfig


DATASET_WRAPPER_MAPPING = {
    "mnist": mnist.MNISTWrapper,
    "cifar10": cifar10.CIFAR10Wrapper,
    "imagenet": imagenet.ImageNetWrapper,
    "ember_2018_2": ember.Ember182Wrapper,
    "ember_2017_2": ember.Ember172Wrapper,
    "ember_2017_1": ember.Ember171Wrapper,
}


def get_dataset_wrapper(data_config: DatasetConfig):
    """
        Create dataset wrapper for given data-config
    """
    wrapper = DATASET_WRAPPER_MAPPING.get(data_config.name, None)
    if not wrapper:
        raise NotImplementedError(f"Dataset {data_config.name} not implemented")
    return wrapper(data_config)
