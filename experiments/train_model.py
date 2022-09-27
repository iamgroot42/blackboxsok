"""
    Script for training models. Useful for victim/attacker models, especially
    for malware-related models.
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import torch as ch

from bbeval.config import ModelTrainingConfig
from bbeval.datasets.utils import get_dataset_wrapper
from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args = parser.parse_args()
    config = ModelTrainingConfig.load(args.config, drop_extra_fields=False)

    train_config = config.train_config
    ds_config = train_config.data_config
    model_config = config.model_config
    batch_size = train_config.batch_size

    # Get model
    model = get_model_wrapper(model_config)
    model.cuda()

    # Get data-loader, make sure it works
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)
    train_loader, val_loader, test_loader = ds.get_loaders(
        batch_size=batch_size)
    # Compute clean accuracy
    loss_function = get_loss_fn("ce")

    # Train model
    def acc_fn(x, y):
        return ch.mean(1.*(x == y)).item()
    kwargs = {
        "loss_function": loss_function,
        "acc_fn": acc_fn,
    }
    model.train(train_loader, val_loader, train_config, **kwargs)

    # TODO: Save
    model.save("testing.pkl")
