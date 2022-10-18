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
    x_sample_adv, queries_used = attacker.attack(x_orig=x_orig, x_adv=x_sample_adv, y_label=y_label, x_target=x_target, y_target=y_target)
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
    batch_size = config.batch_size

    # Get data-loader, make sure it works
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)

    # Extract configs
    attacker_config_1: AttackerConfig = config.first_attack_config()
    attacker_config_2: AttackerConfig = config.second_attack_config()
    #attacker_config_2: AttackerConfig = config.second_attack_config()

    # Load up model(s)
    target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)
    
    #print(target_model_1)
    
    total_queries_used = 0
    correctly_classified = 0
    num_processed = 0
    total_transfered=0
    total_img =100
    batch_size = 10
    loss_function = get_loss_fn("ce")
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    x_target=1

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

    second_attack_x =[]
    while n < total_img:
        x_orig = correct_images[i:i + batch_size]
        y_label = correct_labels[i:i + batch_size]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        correctly_classified += ch.count_nonzero(target_model_prediction == y_label)
        n += batch_size
    print("The clean accuracy is %s %%" % str(float(correctly_classified / total_img*100)))
    #

    second_attack_x=correct_images[0:total_img]
    second_attack_y=correct_labels[0:total_img]

    #print(attacker_config_1.name)
    for i in tqdm(range(int(total_img/batch_size))):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case

        x_orig, y_label = correct_images[i * batch_size:i * batch_size + batch_size], correct_labels[i * batch_size:i * batch_size + batch_size]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()

        num_class = ds.num_classes
        if attacker_config_1.targeted:
            # mode = "easiest"/"hardest"/"random"/"user"
            # mode = attacker_config_1.target_label_selection_mode
            mode = "random"
            y_target = get_target_label(
                mode, x_orig, target_model_1, num_class, y_label, batch_size)
            y_target = y_target.cuda()
        else:
            y_target = y_label
        #print(y_target.shape)
        # Perform attack
        (x_sample_adv, queries_used_1), attacker_1 = single_attack(target_model_1,
                                                                                   aux_models=aux_models_1,
                                                                                   x_orig=x_orig,
                                                                                   x_sample_adv=x_orig,
                                                                                   y_label=y_label,
                                                                                   x_target=x_target,
                                                                                   y_target=y_target,
                                                                                   attacker_config=attacker_config_1,
                                                                                   experiment_config=config)
        attacker_1.save_results()
        
        
        for j in range(len(x_orig)):
            second_attack_x[i * batch_size +j ]=x_sample_adv[j]
            second_attack_y[i * batch_size +j ]=y_label[j]
        
        target_model_1.set_eval()  # Make sure model is in eval model
        target_model_1.zero_grad()  # Make sure no leftover gradients
        target_model_output = target_model_1.forward(x_sample_adv)
        target_model_prediction = ch.max(target_model_output, 1).indices
        if attacker_config_1.targeted:
            total_transfered += ch.count_nonzero(target_model_prediction == y_target)
        else:
            total_transfered += ch.count_nonzero(target_model_prediction != y_target)

    print("total_transfered",total_transfered)

    is_bayes=True
    if is_bayes:
        for i in range(int(1)):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
            second_attack_x=ch.Tensor(second_attack_x)
            print(len(second_attack_x))
            second_attack_x, y_label = second_attack_x.cuda(), y_label.cuda()
            num_class = ds.num_classes
            x_target=second_attack_x
            y_target = y_label
            target_model_2, aux_models_2 = get_model_and_aux_models(attacker_config_2)
            # Perform attack
            (x_sample_adv, queries_used_2), attacker_2 = second_attack(target_model_2,
                                                                                    aux_models=aux_models_2,
                                                                                    x_orig=second_attack_x,
                                                                                    x_sample_adv=second_attack_x,
                                                                                    x_target=x_target,
                                                                                    y_label=second_attack_y,
                                                                                    y_target=y_target,
                                                                                    attacker_config=attacker_config_2,
                                                                                    experiment_config=config)
            total_queries_used += queries_used_2
            attacker_2.save_results()
            print("transferability:", total_transfered)
            if attacker_config_2.targeted:
                total_transfered +=x_sample_adv[-1]
            else:
                total_transfered +=x_sample_adv[-1]

            
            
    else:
        for i in tqdm(range(int(total_img/batch_size))):
            # the original dataset is normalized into the range of [0,1]
            # specific attacks may have different ranges and should be handled case by case
            
            x_orig, y_label = second_attack_x[i * batch_size:i * batch_size + batch_size], correct_labels[i * batch_size:i * batch_size + batch_size]
            x_orig, y_label = x_orig.cuda(), y_label.cuda()

            num_class = ds.num_classes
            if attacker_config_1.targeted:
                # mode = "easiest"/"hardest"/"random"/"user"
                # mode = attacker_config_1.target_label_selection_mode
                mode = "random"
                y_target = get_target_label(
                    mode, x_orig, target_model_1, num_class, y_label, 10)
                y_target = y_target.cuda()
            else:
                y_target = second_attack_y
            target_model_2, aux_models_2 = get_model_and_aux_models(attacker_config_2)
            # Perform attack
            (x_sample_adv, queries_used_2), attacker_2 = single_attack(target_model_2,
                                                                                    aux_models=aux_models_2,
                                                                                    x_orig=second_attack_x,
                                                                                    x_sample_adv=second_attack_x,
                                                                                    x_target=x_target,
                                                                                    y_label=second_attack_y,
                                                                                    y_target=y_target,
                                                                                    attacker_config=attacker_config_2,
                                                                                    experiment_config=config)
            total_queries_used += queries_used_2
            attacker_2.save_results()

            target_model_2.set_eval()  # Make sure model is in eval model
            target_model_2.zero_grad()  # Make sure no leftover gradients
            target_model_output = target_model_2.forward(x_sample_adv)
            target_model_prediction = ch.max(target_model_output, 1).indices
            if attacker_config_2.targeted:
                total_transfered += ch.count_nonzero(target_model_prediction == y_target)
            else:
                total_transfered += ch.count_nonzero(target_model_prediction != second_attack_y)


    '''
    untransfered_x=[]
    y_label=y_label.tolist()
    for y_l in compare_x.keys():
        index_in_orig = y_label.index(y_l)
        untransfered_x.append(x_orig[index_in_orig])
        
    
    y_label = compare_x.keys().cuda()
    print(len(y_label))
    untransfered_x =ch.Tensor(untransfered_x)
    print(len(untransfered_x))

    for i in range(int(1)):
        # the original dataset is normalized into the range of [0,1]
        # specific attacks may have different ranges and should be handled case by case
        num_class = ds.num_classes
        x_target=second_attack_x
        y_target = y_label
        target_model_2, aux_models_2 = get_model_and_aux_models(attacker_config_2)
        # Perform attack
        (x_sample_adv, queries_used_2, compare_x), attacker_2 = second_attack(target_model_2,
                                                                                aux_models=aux_models_2,
                                                                                x_orig=untransfered_x,
                                                                                x_sample_adv=untransfered_x,
                                                                                x_target=x_target,
                                                                                y_label=y_label,
                                                                                y_target=y_target,
                                                                                attacker_config=attacker_config_2,
                                                                                experiment_config=config)

    '''

    transferability = float(total_transfered / total_img*100)
    print("Target model: %s " % (str(attacker_config_2.adv_model_config.name)))
    print("Aux model: %s" % (str(attacker_config_2.aux_model_configs[0].name)))
    print("The transferability of %s is %s %%" % (str(attacker_config_1.name+attacker_config_2.name), str(transferability)))




with open('experiment.txt', 'a') as f:
    f.write('\n')
    f.write('\n')
    f.write("Target model: %s " % (str(attacker_config_1.adv_model_config.name)))
    f.write('\n')
    f.write("Aux model: %s" % (str(attacker_config_1.aux_model_configs[0].name)))
    f.write('\n')
    f.write("The transferbility of %s is %s %%" % (str(attacker_config_1.name + attacker_config_2.name), str(transferability)))
    f.write('\n')
    f.write("--- %s seconds ---" % (time.time() - start_time))
