import torch as ch
from simple_parsing import ArgumentParser
from pathlib import Path
import os

from bbeval.models.pytorch.image import Inceptionv3, ResNet18, VGG16

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
    target_model = Inceptionv3(model_config)
    # target_model = get_model_wrapper(model_config)
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

    # TODO: temporarily testing local mdoels, merge to get_wrapper_function later
    aux_models = {}
    local_model_config = attacker_config.aux_model_config
    local_model = ResNet18(local_model_config)
    local_model.cuda()
    aux_models['resnet18'] = local_model

    x_orig, y_label = next(iter(test_loader))
    y_sample, y_target = next(iter(test_loader))
    # print("x sample")
    # print(x_sample)
    x_orig, y_label, y_target = x_orig.cuda(), y_label.cuda(), y_target.cuda()
    x_adv = x_orig

    attacker = get_attack_wrapper(target_model, aux_models, attacker_config)
    x_sample_adv, queries_used = attacker.attack(x_orig, y_label, y_target, x_adv)
    attacker.save_results()

    print("%s attack is completed" % attacker_config.name)
