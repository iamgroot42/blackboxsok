from unittest import skip
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


def single_attack(target_model, aux_models, x_orig, x_sample_adv, y_label, y_target, attacker_config: AttackerConfig,
                  experiment_config: ExperimentConfig):
    attacker = get_attack_wrapper(target_model, aux_models, attacker_config, experiment_config)
    x_sample_adv, queries_used, num_transfered = attacker.attack(x_orig, x_sample_adv, y_label, y_target)
    return (x_sample_adv, queries_used, num_transfered), attacker


# os.environ['TORCH_HOME'] = '/p/blackboxsok/models/imagenet_torch' # download imagenet models to project directory
if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)

    ds_config = config.dataset_config
    # batch_size = config.batch_size
    batch_size = 5

    num_img =5

    # Get data-loader, make sure it works
    ds: CustomDatasetWrapper = get_dataset_wrapper(ds_config)
    _, _, test_loader = ds.get_loaders(
        batch_size=batch_size, eval_shuffle=True)

    loss_function = get_loss_fn("ce")

    # Extract configs
    attacker_config_1: AttackerConfig = config.first_attack_config()
    attacker_config_2: AttackerConfig = config.second_attack_config()

    # Load up model(s)
    target_model_1, aux_models_1 = get_model_and_aux_models(attacker_config_1)
    total_queries_used = 0
    total_transfered = 0
    num_processed = 0
    correct_images = []
    correct_labels = []

    start_time = time.time()
    target_model_1.set_eval()  # Make sure model is in eval model
    target_model_1.zero_grad()  # Make sure no leftover gradients

    
        
    counter=0
    # select correctly classfied images
    while len(correct_images) <= num_img:
        print(len(correct_images))
        x_orig, y_label = next(iter(test_loader))
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        for i in range(len(target_model_prediction)):
            if target_model_prediction[i] == y_label[i] and ((y_label[i] not in correct_labels) or counter>10000):
                counter = 0
                correct_images.append(x_orig[i].detach().cpu().numpy())
                correct_labels.append(int(y_label[i].detach().cpu()))
            else:
                counter+=1
        if len(correct_images) == num_img:
            break
    print("--- %s seconds ---" % (time.time() - start_time))
    for i in range(num_img+1):
        if i not in correct_labels:
            print(i)


    print("==========")


    if attacker_config_1.targeted:
        counter =0
        y_target =[]
        # mode = "easiest"/"hardest"/"random"/"user"
        mode = "random"
        y_target = get_target_label(mode, correct_images, target_model_1, num_img, correct_labels, num_img)
        print(len(y_target))

        x_target=[]
        for i in range(num_img):
            x_target.append(-1)

        target_l = []
        counter=0

        while len(target_l)<= num_img:
            print("len(target_l):",len(target_l))
            x_orig, y_label = next(iter(test_loader))
            x_orig, y_label = x_orig.cuda(), y_label.cuda()
            target_model_output = target_model_1.forward(x_orig)
            target_model_prediction = ch.max(target_model_output, 1).indices
            for i in range(len(target_model_prediction)):
                cur=int(target_model_prediction[i].detach().cpu())
                try:
                    y_ind = correct_labels.index(cur)
                    if  target_model_prediction[i] == y_label[i] and cur not in target_l:
                        x_target[y_ind]=x_orig[i].detach().cpu().numpy()
                        target_l.append(cur)
                        print(cur)
                except:
                    counter+=1
                    continue
            if len(target_l)==num_img:
                break
        print(x_target)     
    correct_labels = ch.Tensor(correct_labels)
    correct_labels = correct_labels.type(ch.LongTensor)
    correct_images = ch.Tensor(correct_images)
    # check whether images are correctly classified
    i = 0
    while i < num_img:
        x_orig = correct_images[i:i + batch_size]
        y_label = correct_labels[i:i + batch_size]
        x_orig, y_label = x_orig.cuda(), y_label.cuda()
        target_model_output = target_model_1.forward(x_orig)
        target_model_prediction = ch.max(target_model_output, 1).indices
        total_transfered += ch.count_nonzero(target_model_prediction == y_label)
        i += batch_size
        #print(total_transfered)
    print(total_transfered)
    # print(correct_images.shape)

    y_target = y_target.type(ch.float)
    y_target = ch.Tensor(y_target)
    x_target = ch.Tensor(x_target)


    print("===============")
    print("y_target:",y_target.shape)
    print("x_target:",x_target.shape)



    model_name=attacker_config_1.adv_model_config.name

    ch.save(correct_images, 'data/'+model_name+'/correct_images.pt')
    ch.save(correct_labels, 'data/'+model_name+'/correct_labels.pt')
    ch.save(y_target, 'data/'+model_name+'/y_target.pt')
    ch.save(x_target, 'data/'+model_name+'/x_target.pt')


