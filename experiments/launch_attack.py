import torch as ch
from simple_parsing import ArgumentParser
from pathlib import Path
import os

from bbeval.models.pytorch.image import Inceptionv3
from bbeval.config import AttackerConfig
from bbeval.datasets.utils import get_dataset_wrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.loss import get_loss_fn

# os.environ['TORCH_HOME'] = '/p/blackboxsok/models/imagenet_torch' # download imagenet models to project directory
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

    # TODO: Implement getters that get the right model

    # Get a pretrained ImageNet model
    model = Inceptionv3(model_config)
    model.cuda()
    # Get data-loader, make sure it works
    ds = get_dataset_wrapper(ds_config)

    # _, _, test_loader = ds.get_loaders(batch_size=256, eval_shuffle=True)
    _, _, test_loader = ds.get_loaders(batch_size=32, eval_shuffle=True)
    # Compute clean accuracy
    loss_function = get_loss_fn("ce")
    def acc_fn(predicted, true):
        return ch.mean(1. * (predicted == true))
    
    # eval_loss, eval_acc = model.eval(test_loader, loss_function, acc_fn)
    # print("Clean accuracy: {:.2f}".format(eval_acc))
    # exit()

    x_sample, y_sample = next(iter(test_loader))
    x_sample, y_sample = x_sample.cuda(), y_sample.cuda()
    attacker = get_attack_wrapper(model, attacker_config)
    x_sample_adv, queries_used = attacker.attack(x_sample, y_sample, eps=1.0)
    # attacker.save_results()
