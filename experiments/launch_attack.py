import torch as ch
from simple_parsing import ArgumentParser
from pathlib import Path
import os

from bbeval.config import AttackerConfig
from bbeval.datasets.utils import get_dataset_wrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import MODEL_WRAPPER_MAPPING 
from bbeval.models.utils import get_model_wrapper
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
    # Get a pretrained ImageNet model
    target_model = get_model_wrapper(model_config)
    target_model.cuda()

    ds_config = attacker_config.dataset_config
    # Get data-loader, make sure it works
    ds = get_dataset_wrapper(ds_config)
    _, _, test_loader = ds.get_loaders(batch_size=32, eval_shuffle=True)
    # _, _, test_loader = ds.get_loaders(batch_size=32, eval_shuffle=True)
    # Compute clean accuracy
    loss_function = get_loss_fn("ce")
    def acc_fn(predicted, true):
        return ch.mean(1. * (predicted == true))

    if attacker_config.aux_model_configs:
        aux_models =  get_model_wrapper(attacker_config.aux_model_configs)
        for _key in aux_models:
            aux_models[_key].cuda()
    else:
        aux_models = {}

    # the original dataset is normalized into the range of [0,1]
    # specific attacks may have different ranges and should be handled case by case
    x_orig, y_label = next(iter(test_loader))
    x_orig = x_orig.cuda()
    y_label = y_label.cuda()
    # y_sample, y_target = next(iter(test_loader))

    # x_orig, y_label, y_target = x_orig.cuda(), y_label.cuda(), y_target.cuda()
    x_adv_loc = x_orig
    y_target = y_label

    attacker = get_attack_wrapper(target_model, aux_models, attacker_config)
    x_sample_adv, queries_used = attacker.attack(x_orig, x_adv_loc, y_label, y_target)
    attacker.save_results()

    print("%s attack is completed" % attacker_config.name)
