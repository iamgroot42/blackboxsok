from simple_parsing import ArgumentParser
from pathlib import Path
import torch as ch

from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.datasets.utils import get_dataset_wrapper, get_target_label
from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn


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
    batch_size = 10

    # Get data-loader, make sure it works
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)
    _, _, test_loader = ds.get_loaders(
        batch_size=batch_size, eval_shuffle=True)
    # Compute clean accuracy
    loss_function = get_loss_fn("ce")

    num_correct=0

    # Extract configs
    attacker_config_1: AttackerConfig = config.first_attack_config()
    attacker_config_2: AttackerConfig = config.second_attack_config()

    # Load up model(s)
    target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    for i in range(100):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        print(i)
        x_orig, y_label = next(iter(test_loader))
        x_orig, y_label = x_orig.cuda(), y_label.cuda()

        # outputs the accuracy of the target model
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        num_correct += ch.count_nonzero(target_model_prediction == y_label)

    accuracy = float(num_correct / 1000) * 100
    print("The accuracy of the target model is %s %%" % str(accuracy))

        # for name, model in aux_models_1.items():
        #     model.set_eval()  # Make sure model is in eval model
        #     model.zero_grad()  # Make sure no leftover gradients
        #     model_output = model.forward(x_orig)
        #     model_prediction = ch.max(model_output, 1).indices
        #     batch_size = len(y_target)
        #     num_correct = ch.count_nonzero(model_prediction == y_label)
        #     accuracy = float(num_correct / batch_size) * 100
        #     print("The accuracy of %s is %s %%" % (name,str(accuracy)))