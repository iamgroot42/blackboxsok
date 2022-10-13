import random
import torch as ch
from tqdm import tqdm

from bbeval.datasets.image import mnist, cifar10, imagenet
from bbeval.datasets.malware import bodmas_dataset, ember, ember_ucsb_combined
from bbeval.config import DatasetConfig


DATASET_WRAPPER_MAPPING = {
    "mnist": mnist.MNISTWrapper,
    "cifar10": cifar10.CIFAR10Wrapper,
    "imagenet": imagenet.ImageNetWrapper,
    "ember_2018_2": ember.Ember182Wrapper,
    "ember_2017_2": ember.Ember172Wrapper,
    "ember_2017_1": ember.Ember171Wrapper,
    "bodmas": bodmas_dataset.BODMASWrapper,
    "ember": ember_ucsb_combined.EmberCombinedWrapper,
    "ucsb": ember_ucsb_combined.UCSBWrapper
}


def get_dataset_wrapper(data_config: DatasetConfig):
    """
        Create dataset wrapper for given data-config
    """
    wrapper = DATASET_WRAPPER_MAPPING.get(data_config.name, None)
    if not wrapper:
        raise NotImplementedError(f"Dataset {data_config.name} not found")
    return wrapper(data_config)


def rand_int_gen_exclu(num_min, num_max, num_exclu, res_len):
    tmp = []
    for j in range(res_len):
        tmp.append(random.choice(
            [i for i in range(num_min, num_max) if i != num_exclu[j]]))
    tmp = ch.tensor(tmp)
    return tmp


def get_target_label(mode, x_orig, model, num_class, y_label, batch_size):
    if mode == "easiest":
        target_model_output = model.forward(x_orig)
        target_label = ch.kthvalue(target_model_output, num_class-1).indices
    if mode == "hardest":
        target_model_output = model.forward(x_orig)
        target_label = ch.min(target_model_output, 1).indices
    if mode == "random":
        target_label = rand_int_gen_exclu(
            0, num_class - 1, y_label, batch_size)
    if mode == "user":
        target_class = int(
            input('Enter your target class from %s to %s: ' % (0, num_class - 1)))
        target_label = [target_class] * batch_size
        target_label = ch.tensor(target_label)
    return target_label


def collect_data_from_loader(loader):
    X, Y = [], []
    for batch in tqdm(loader, desc="Collecting data from loader"):
        X.append(batch[0])
        Y.append(batch[1])
    X = ch.cat(X, dim=0)
    Y = ch.cat(Y, dim=0)
    return X, Y
