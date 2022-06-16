import torch as ch
from simple_parsing import ArgumentParser
from pathlib import Path
import os
import random

from bbeval.config import AttackerConfig
from bbeval.datasets.utils import get_dataset_wrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import MODEL_WRAPPER_MAPPING
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn

# os.environ['TORCH_HOME'] = '/p/blackboxsok/models/imagenet_torch' # download imagenet models to project directory
if __name__ == "__main__":
    def rand_int_gen_exclu(num_min, num_max, num_exclu, res_len):
        tmp = []
        for j in range(res_len):
            tmp.append(random.choice([i for i in range(num_min, num_max) if i != num_exclu[j]]))
        tmp = ch.tensor(tmp)
        return tmp


    # test through colab and put to the right place
    def get_target_label(mode, x_orig, model, num_class, y_label, batch_size):
        if mode == "easiest":
            target_model_output = model.forward(x_orig)
            target_label = ch.kthvalue(target_model_output, num_class).indices
        if mode == "hardest":
            target_model_output = model.forward(x_orig)
            target_label = ch.min(target_model_output, 1).indices
        if mode == "random":
            target_label = rand_int_gen_exclu(0, num_class - 1, y_label, batch_size)
        return target_label


    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config1", help="Specify config file", type=Path)
    parser.add_argument(
        "--config2", help="Specify config file", type=Path)

    args, remaining_argv = parser.parse_known_args()
    if args.config2 == None:
        # Attempt to extract as much information from config file as you can
        config1 = AttackerConfig.load(args.config1, drop_extra_fields=False)
        # Also give user the option to provide config values over CLI
        parser1 = ArgumentParser(parents=[parser])
        parser1.add_arguments(AttackerConfig, dest="attacker_config1", default=config1)
        args1 = parser1.parse_args(remaining_argv)
        # Extract relevant configs
        attacker_config1: AttackerConfig = args1.attacker_config1
        model_config1 = attacker_config1.adv_model_config
        # Get a pretrained ImageNet model
        target_model1 = get_model_wrapper(model_config1)
        target_model1.cuda()

        batch_size = 32
        ds_config = attacker_config1.dataset_config
        # Get data-loader, make sure it works
        ds = get_dataset_wrapper(ds_config)
        _, _, test_loader = ds.get_loaders(batch_size=batch_size, eval_shuffle=True)
        # Compute clean accuracy
        loss_function = get_loss_fn("ce")

        if attacker_config1.aux_model_configs:
            aux_models1 = get_model_wrapper(attacker_config1.aux_model_configs)
            for _key in aux_models1:
                aux_models1[_key].cuda()
        else:
            aux_models1 = {}

        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        x_orig, y_label = next(iter(test_loader))
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        # mode = "easiest"/"hardest"/"random"
        mode = "hardest"
        num_class = 1000
        y_target = get_target_label(mode, x_orig, target_model1, num_class, y_label, batch_size)
        y_target = y_target.cuda()
        attacker1 = get_attack_wrapper(target_model1, aux_models1, attacker_config1)
        x_sample_adv1, queries_used1 = attacker1.attack(x_orig, y_label, y_target, x_orig)
        attacker1.save_results()

        print("%s attack is completed" % attacker_config1.name)

    else:
        # Attempt to extract as much information from config file as you can
        config1 = AttackerConfig.load(args.config1, drop_extra_fields=False)
        config2 = AttackerConfig.load(args.config2, drop_extra_fields=False)
        # Also give user the option to provide config values over CLI
        parser1 = ArgumentParser(parents=[parser])
        parser1.add_arguments(AttackerConfig, dest="attacker_config1", default=config1)
        args1 = parser1.parse_args(remaining_argv)
        parser2 = ArgumentParser(parents=[parser])
        parser2.add_arguments(AttackerConfig, dest="attacker_config2", default=config2)
        args2 = parser2.parse_args(remaining_argv)
        # Extract relevant configs
        attacker_config1: AttackerConfig = args1.attacker_config1
        model_config1 = attacker_config1.adv_model_config
        attacker_config2: AttackerConfig = args2.attacker_config2
        model_config2 = attacker_config1.adv_model_config
        # Get a pretrained ImageNet model
        target_model1 = get_model_wrapper(model_config1)
        target_model1.cuda()
        target_model2 = get_model_wrapper(model_config2)
        target_model2.cuda()

        ds_config = attacker_config1.dataset_config
        # Get data-loader, make sure it works
        ds = get_dataset_wrapper(ds_config)
        _, _, test_loader = ds.get_loaders(batch_size=32, eval_shuffle=True)
        # Compute clean accuracy
        loss_function = get_loss_fn("ce")

        if attacker_config1.aux_model_configs:
            aux_models1 = get_model_wrapper(attacker_config1.aux_model_configs)
            for _key in aux_models1:
                aux_models1[_key].cuda()
        else:
            aux_models1 = {}

        if attacker_config2.aux_model_configs:
            aux_models2 = get_model_wrapper(attacker_config2.aux_model_configs)
            for _key in aux_models2:
                aux_models2[_key].cuda()
        else:
            aux_models2 = {}

        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        x_orig, y_label = next(iter(test_loader))
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        # mode = "easiest"/"hardest"/"random"
        mode = "random"
        num_class = 1000
        y_target = get_target_label(mode, x_orig, target_model1, num_class, y_label, batch_size)
        y_target = y_target.cuda()

        attacker1 = get_attack_wrapper(target_model1, aux_models1, attacker_config1)
        attacker2 = get_attack_wrapper(target_model2, aux_models2, attacker_config2)
        x_sample_adv1, queries_used1 = attacker1.attack(x_orig, y_label, y_target, x_orig)
        attacker1.save_results()
        x_sample_adv2, queries_used2 = attacker2.attack(x_orig, y_label, y_target, x_sample_adv1)
        attacker2.save_results()

        print("%s attack and %s attack is completed" % (attacker_config1.name, attacker_config2.name))
