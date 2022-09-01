from simple_parsing import ArgumentParser
from pathlib import Path

from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.datasets.utils import get_dataset_wrapper, get_target_label
from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn
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


def single_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, x_target, y_target, attacker_config: AttackerConfig,
                  experiment_config: ExperimentConfig):
    attacker = get_attack_wrapper(target_model, aux_models, attacker_config, experiment_config)
    x_sample_adv, queries_used = attacker.attack(x_orig, x_sample_adv, y_label, y_target)
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
    total_transfered=0
    batch_size = 10
    loss_function = get_loss_fn("ce")
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    ds_config = config.dataset_config
    if ds_config.name == 'imagenet':
        if attacker_config_1.adv_model_config.name == 'inceptionv3':
            correct_images = ch.load('data/inceptionv3/correct_images.pt')
            correct_labels = ch.load('data/inceptionv3/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'inceptionv4':
            correct_images = ch.load('data/inceptionv4/correct_images.pt')
            correct_labels = ch.load('data/inceptionv4/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'inceptionresnetv2':
            correct_images = ch.load('data/inceptionresnetv2/correct_images.pt')
            correct_labels = ch.load('data/inceptionresnetv2/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'resnet101':
            correct_images = ch.load('data/resnet101/correct_images.pt')
            correct_labels = ch.load('data/resnet101/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'ensinceptionresnetv2':
            correct_images = ch.load('data/ensinceptionresnetv2/correct_images.pt')
            correct_labels = ch.load('data/ensinceptionresnetv2/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'advinceptionv3':
            correct_images = ch.load('data/advinceptionv3/correct_images.pt')
            correct_labels = ch.load('data/advinceptionv3/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'resnet50':
            correct_images = ch.load('data/resnet50/correct_images.pt')
            correct_labels = ch.load('data/resnet50/correct_labels.pt')
        if attacker_config_1.adv_model_config.name == 'vgg16_bn':
            correct_images = ch.load('data/vgg16_bn/correct_images.pt')
            correct_labels = ch.load('data/vgg16_bn/correct_labels.pt')
    # start_time = time.time()

    n, i = 0, 0
    start_time = time.time()

    while n < 1000:
        x_orig = correct_images[i:i + 10]
        y_label = correct_labels[i:i + 10]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        correctly_classified += ch.count_nonzero(target_model_prediction == y_label)
        n += 10
    print("The clean accuracy is %s %%" % str(float(correctly_classified / 10)))
    #
    for i in range(int(1)):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case

        x_orig, y_label = correct_images[i * 100:i * 100 + 100], correct_labels[i * 100:i * 100 + 100]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()

        num_class = ds.num_classes
        if attacker_config_1.targeted:
            x_target = ch.load('data/resnet50/x_target.pt')
            y_target = ch.load('data/resnet50/y_target.pt')
        else:
            x_target=x_orig
            y_target = y_label

        # Perform attack
        (x_sample_adv, queries_used_1), attacker_1 = single_attack(target_model_1,
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

        target_model_1.set_eval()  # Make sure model is in eval model
        target_model_1.zero_grad()  # Make sure no leftover gradients
        target_model_output = target_model_1.forward(x_sample_adv)
        target_model_prediction = ch.max(target_model_output, 1).indices
        if attacker_config_1.targeted:
            total_transfered += ch.count_nonzero(target_model_prediction == y_target)
        else:
            total_transfered += ch.count_nonzero(target_model_prediction != y_target)
'''
    transferability = float(total_transfered / 10)
    print("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
    print("Aux model: %s" % (str(attacker_config_1.aux_model_configs[0].name)))
    print("The transferability of %s is %s %%" % (str(attacker_config_1.name), str(transferability)))

with open('experiment.txt', 'a') as f:
    f.write('\n')
    f.write('\n')
    f.write("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
    f.write('\n')
    f.write("Aux model: %s" % (str(attacker_config_1.aux_model_configs[0].name)))
    f.write('\n')
    f.write("The transferbility of %s is %s %%" % (str(attacker_config_1.name), str(transferability)))
    f.write('\n')
    f.write("--- %s seconds ---" % (time.time() - start_time))
'''