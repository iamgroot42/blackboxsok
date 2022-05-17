import torch as ch
from bbeval.attacker.top1_score import RayS
from bbeval.datasets.image.cifar10 import CIFAR10Wrapper
from bbeval.models.pytorch.image import Inceptionv3
from bbeval.config import AttackerConfig
from simple_parsing import ArgumentParser
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = AttackerConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(AttackerConfig, dest="attacker_config", default=config)
    args = parser.parse_args(remaining_argv)

    # Extract relevant configs
    attacker_config: AttackerConfig = args.attacker_config
    model_config = attacker_config.adv_model_config
    ds_config = attacker_config.dataset_config

    # TODO: Implement getters that get the right model, attack, dataset
    # Using names provided

    # Get a pretrained ImageNet model
    model = Inceptionv3(model_config)
    model.cuda()
    # Get data-loader, make sure it works
    ds = CIFAR10Wrapper(ds_config)
    _, _, test_loader = ds.get_loaders(batch_size=32)
    x_sample, y_sample = next(iter(test_loader))
    x_sample, y_sample = x_sample.cuda(), y_sample.cuda()
    # For now, make a random image
    x_sample = ch.rand(32, 3, 299, 299).cuda()
    attacker = RayS(model, attacker_config)
    x_sample_adv, queries_used = attacker.attack(x_sample, y_sample, eps=1.0)
    attacker.save_results()
