from simple_parsing import ArgumentParser
from pathlib import Path
import torch as ch

from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.datasets.utils import get_dataset_wrapper, get_target_label
from bbeval.datasets.base import CustomDatasetWrapper
from bbeval.attacker.utils import get_attack_wrapper
from bbeval.models.utils import get_model_wrapper
from bbeval.loss import get_loss_fn

from torch.profiler import profile, record_function, ProfilerActivity


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


def single_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, y_target,
                  attacker_config: AttackerConfig, experiment_config: ExperimentConfig,
                  profiler = None, suffix: str = ""):
    if profiler:
        with record_function("get_attack_wrapper" + suffix):
            attacker = get_attack_wrapper(target_model, aux_models, attacker_config, experiment_config)
        with record_function("attack" + suffix):
            x_sample_adv, queries_used = attacker.attack(x_orig, x_sample_adv, y_label, y_target)
    else:
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
    batch_size = config.batch_size

    # Get data-loader, make sure it works
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)
    _, _, test_loader = ds.get_loaders(
        batch_size=batch_size, eval_shuffle=True)
    # Compute clean accuracy
    loss_function = get_loss_fn("ce")

    # the original dataset is normalized into the range of [0,1]
    # specific attacks may have different ranges and should be handled case by case
    x_orig, y_label = next(iter(test_loader))
    x_orig, y_label = x_orig.cuda(), y_label.cuda()

    # Extract configs
    attacker_config_1: AttackerConfig = config.first_attack_config()
    attacker_config_2: AttackerConfig = config.second_attack_config()

    def execute(profiler = None):
        # Load up model(s)
        if profiler:
            with record_function("load_model_1"):
                target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)
        else:
            target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)

        num_class = ds.num_classes
        if attacker_config_1.targeted:
            # mode = "easiest"/"hardest"/"random"/"user"
            # mode = attacker_config_1.target_label_selection_mode
            mode = "random"
            if profiler:
                with record_function("target_label"):
                    y_target = get_target_label(
                        mode, x_orig, target_model_1, num_class, y_label, batch_size)
            else:
                y_target = get_target_label(
                    mode, x_orig, target_model_1, num_class, y_label, batch_size)
            y_target = y_target.cuda()
        else:
            y_target = y_label

        # Perform attack
        total_queries_used = 0
        (x_sample_adv, queries_used_1), attacker_1 = single_attack(
                target_model_1,
                aux_models=aux_models_1,
                x_orig=x_orig,
                x_sample_adv=x_orig,
                y_label=y_label,
                y_target=y_target,
                attacker_config=attacker_config_1,
                experiment_config=config,
                profiler=profiler,
                suffix="_1")
        profiler.step()
        total_queries_used += queries_used_1
        attacker_1.save_results()

        # Follow up attack if config provided
        if attacker_config_2:
            if profiler:
                with record_function("load_model_2"):
                    target_model_2, aux_models_2 = get_model_and_aux_models(attacker_config_2)
            else:
                target_model_2, aux_models_2 = get_model_and_aux_models(attacker_config_2)
            (x_sample_adv, queries_used_2), attacker_2 = single_attack(target_model_2,
                                                                       aux_models=aux_models_2,
                                                                       x_orig=x_orig,
                                                                       x_sample_adv=x_sample_adv,
                                                                       y_label=y_label,
                                                                       y_target=y_target,
                                                                       attacker_config=attacker_config_2,
                                                                       experiment_config=config,
                                                                       profiler=profiler,
                                                                       suffix="_2")
            total_queries_used += queries_used_2
            attacker_2.save_results()

    if config.profiler:
        # Profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # TODO: Switch to below (with adjusted values) when profiler is sent in for per-example tracking
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=ch.profiler.schedule(wait=2, warmup=2,active=6, repeat=1)) as prof:
            execute(prof)
        
            # Print profiler statistics
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        # Save (dump) profiler statistics in json
        # prof.export_chrome_trace("trace.json")

    else:
        # Execute
        execute()
