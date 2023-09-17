from simple_parsing import ArgumentParser
from pathlib import Path
import os

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


def single_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, x_target, y_target,
                  attacker_config: AttackerConfig,
                  experiment_config: ExperimentConfig,experiment_file_name):
    attacker = get_attack_wrapper(target_model, aux_models, attacker_config, experiment_config)
    x_sample_adv, queries_used = attacker.attack(x_orig=x_orig, x_adv=x_sample_adv, y_label=y_label, x_target=x_target,
                                                 y_target=y_target,experiment_file_name=experiment_file_name)
    return (x_sample_adv, queries_used), attacker


# os.environ['TORCH_HOME'] = '/project/uvasrg_paid/blackboxsok/models/imagenet_torch' # download imagenet models to project directory
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
    total_transfered = 0
    batch_size = 10
    loss_function = get_loss_fn("ce")
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    # Hard/random label model
    # mode_prefix = "hard_"
    mode_prefix = ""

    ds_config = config.dataset_config
    target_model_name = mode_prefix + attacker_config_1.adv_model_config.name
    # / p / blackboxsok / experiment / data / hard_
    base_path = get_cache_dir_path() #  / p / blackboxsok / experiment
    correct_images_path = os.path.join(base_path, 'data/') + target_model_name + '/correct_images.pt'
    correct_labels_path = os.path.join(base_path, 'data/') + target_model_name + '/correct_labels.pt'
    target_images_path =  os.path.join(base_path, 'data/') + target_model_name + '/target_images.pt'
    target_labels_path =  os.path.join(base_path, 'data/') + target_model_name + '/target_labels.pt'
    try:
        correct_images = ch.load(correct_images_path)
        correct_labels = ch.load(correct_labels_path)
        if attacker_config_1.targeted:
            target_images = ch.load(target_images_path)
            target_labels = ch.load(target_labels_path)
            target_labels = target_labels.type(ch.LongTensor)

        else:
            target_images = correct_images
            target_labels = correct_labels
    except:
        raise NotImplementedError(f"The image of {target_model_name} is not saved yet")

    if attacker_config_1.targeted:
        prefix = "target"
        iterations=100
    else:
        prefix = "untarget"
        iterations=10

    experiment_file_name = 'result/' + attacker_config_1.name + '_' + prefix + '_' + target_model_name + '_eps' + str(
        int(attacker_config_1.eps)) + '.txt'

    n=0
    while n < 1000:
        x_orig = correct_images[n:n + 10]
        y_label = correct_labels[n:n + 10]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        # print(target_model_prediction)
        # break
        correctly_classified += ch.count_nonzero(target_model_prediction == y_label)
        n += 10
    print("The clean accuracy is %s %%" % str(float(correctly_classified / 10)))
    #
    batch_size = 10 # 10
    n_iters = 10 # 10
    assert batch_size * n_iters == 100, "batch_size * n_iters should be 100"


    with open(experiment_file_name, 'a') as f:
        f.write("epsilon: %s" % (str(attacker_config_1.eps)))
        f.write('\n')
        f.write("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
        f.write('\n')
        f.write("batch size: %s" % (str(10)))
        f.write('\n')
        f.write("number of iteration: %s" % (str(iterations)))
        f.write('\n')

    for i in tqdm(range(int(n_iters))):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        with open(experiment_file_name, 'a') as f:
            f.write("batch: %s" % (str(i)))
            f.write('\n')

        x_orig, y_label = correct_images[i * batch_size:i * batch_size + batch_size], correct_labels[i * batch_size:i * batch_size + batch_size]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        x_target, y_target = target_images[i * batch_size:i * batch_size + batch_size], target_labels[i * batch_size:i * batch_size + batch_size]
        x_target, y_target = x_target.cuda(), y_target.cuda()
        num_class = ds.num_classes

        # print(x_orig[:2])
        # print(x_target[:2])
        # print(y_label[:2])
        # print(y_target[:2])
        # exit(0)

        # Perform attack
        (x_sample_adv, queries_used_1), attacker_1 = single_attack(target_model_1,
                                                                   aux_models=aux_models_1,
                                                                   x_orig=x_orig,
                                                                   x_sample_adv=x_orig,
                                                                   y_label=y_label,
                                                                   x_target=x_target,
                                                                   y_target=y_target,
                                                                   attacker_config=attacker_config_1,
                                                                   experiment_config=config,
                                                                   experiment_file_name=experiment_file_name)
        total_queries_used += queries_used_1
        attacker_1.save_results()
        x_sample_adv = x_sample_adv.to(device='cuda', dtype=ch.float)
        target_model_1.set_eval()  # Make sure model is in eval model
        target_model_1.zero_grad()  # Make sure no leftover gradients
        target_model_output = target_model_1.forward(x_sample_adv)
        target_model_prediction = ch.max(target_model_output, 1).indices
        if attacker_config_1.targeted:
            total_transfered += ch.count_nonzero(target_model_prediction == y_target)
        else:
            total_transfered += ch.count_nonzero(target_model_prediction != y_target)

    transferability = float(total_transfered / 1)
    print("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
    print("Aux model: %s" % (str(attacker_config_1.aux_model_configs[0].name)))
    print("The transferability of %s is %s %%" % (str(attacker_config_1.name), str(transferability)))
