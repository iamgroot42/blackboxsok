from simple_parsing import ArgumentParser
import os
from pathlib import Path
from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.datasets.utils import get_dataset_wrapper, get_target_label
from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn
from bbeval.utils import get_cache_dir_path
from tqdm import tqdm
import time
import torch as ch

ch.manual_seed(2)

image_attack = 100

def get_model_and_aux_models(attacker_config: AttackerConfig):
    model_config = attacker_config.adv_model_config
    # Get first model
    target_model = get_model_wrapper(model_config)
    target_model.cuda()

    if attacker_config.aux_model_configs:
        aux_models = get_model_wrapper(attacker_config.aux_model_configs)
        for _key in aux_models:
            aux_models[_key].cuda()
    else:
        aux_models = {}
    return target_model, aux_models


def single_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, y_target, attacker_config: AttackerConfig, experiment_config: ExperimentConfig):
    attacker = get_attack_wrapper( target_model, aux_models, attacker_config, experiment_config)
    x_sample_adv, queries_used = attacker.attack(x_orig, x_sample_adv, y_label,x_target, y_target)
    return (x_sample_adv, queries_used), attacker

def second_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, x_target, y_target, attacker_config: AttackerConfig,
                  experiment_config: ExperimentConfig):
    attacker = get_attack_wrapper(target_model, aux_models, attacker_config, experiment_config)
    x_sample_adv, queries_used = attacker.attack(x_orig=x_orig, x_adv=x_sample_adv, y_label=y_label, x_target=x_target, y_target=y_target)
    return (x_sample_adv, queries_used), attacker

# os.environ['TORCH_HOME'] = '/p/blackboxsok/models/imagenet_torch' # download imagenet models to project directory
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    ds_config = config.dataset_config
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)

    # Extract configs
    attacker_config_1: AttackerConfig = config.first_attack_config()
    attacker_config_2: AttackerConfig = config.second_attack_config()

    # Load up model(s)
    target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)
    total_queries_used = 0
    correctly_classified = 0
    num_processed = 0
    total_success=0
    batch_size = 10
    loss_function = get_loss_fn("ce")
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    ds_config = config.dataset_config
    target_model_name = attacker_config_1.adv_model_config.name
    base_path = get_cache_dir_path() #  / p / blackboxsok / experiment
    correct_images_path = os.path.join(base_path, 'data/') + target_model_name + '/correct_images.pt'
    correct_labels_path = os.path.join(base_path, 'data/') + target_model_name + '/correct_labels.pt'
    target_images_path =  os.path.join(base_path, 'data/') + target_model_name + '/target_images.pt'
    target_labels_path =  os.path.join(base_path, 'data/') + target_model_name + '/target_labels.pt'
    try:
        correct_images = ch.load(correct_images_path)
        correct_labels = ch.load(correct_labels_path)
        target_images = correct_images
        target_labels = correct_labels
        if attacker_config_1.targeted:
            target_images = ch.load(target_images_path)
            target_labels = ch.load(target_labels_path)
            print("hello")
        else:
            target_images = correct_images
            target_labels = correct_labels
    except:
        raise NotImplementedError(f"The image of {target_modal_name} is not saved yet")

    n = 0
    start_time = time.time()

    while n < image_attack:
        x_orig = correct_images[n:n + 10]
        y_label = correct_labels[n:n + 10]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        correctly_classified += ch.count_nonzero(target_model_prediction == y_label)
        n += 10
    print("The clean accuracy is %s %%" % str(float(correctly_classified / 1)))
    #
    for i in range(int(1)):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        x_orig, y_label = correct_images[0:image_attack], correct_labels[0:image_attack]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        x_target, y_target = target_images[0:image_attack], target_labels[0:image_attack]
        x_target, y_target = x_target.cuda(), y_target.cuda()
        num_class = ds.num_classes

        # Perform attack
        (x_sample_adv, queries_used_1), attacker_1 = second_attack(target_model_1,
                                                                                    aux_models=aux_models_1,
                                                                                    x_orig=x_orig,
                                                                                    x_sample_adv=x_orig,
                                                                                    x_target=x_target,
                                                                                    y_label=y_label,
                                                                                    y_target=y_target,
                                                                                    attacker_config=attacker_config_1,
                                                                                    experiment_config=config)
        total_queries_used += queries_used_1
        attacker_1.save_results()
        print("result saved correctly!")
        x_sample_adv = x_sample_adv.to(device='cuda', dtype=ch.float)
        target_model_1.set_eval()  # Make sure model is in eval model
        target_model_1.zero_grad()  # Make sure no leftover gradients
        for idx in range(len(x_sample_adv)):
            x = x_sample_adv[idx].reshape(1,3,299,299)
            x=ch.tensor(x)
            x = x.to(device='cuda', dtype=ch.float)
            target_model_output = target_model_1.forward(x)
            target_model_prediction = ch.max(target_model_output, 1).indices
            if attacker_config_1.targeted:
                total_success += ch.count_nonzero(target_model_prediction == y_target[idx])
            else:
                total_success += ch.count_nonzero(target_model_prediction != y_target[idx])

    success_rate = float(total_success / 1)
    print("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
    print("The success rate of %s is %s %%" % (str(attacker_config_1.name), str(success_rate)))