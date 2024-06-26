import numpy as np
import torch as ch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_gradient import torch_staircase_sign, project_noise, gkern, \
    project_kern
from bbeval.attacker.transfer_methods._manipulate_input import ensemble_input_diversity, input_diversity, clip_by_tensor

import time
import gc

np.set_printoptions(precision=5, suppress=True)


class ODS(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = TransferredAttackConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = get_loss_fn("ce")
        self.norm = None

    def attack(self, x_orig, x_adv=None, y_label=None, x_target=None, y_target=None, experiment_file_name=None):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        eps = self.eps / 255.0
        ODI_step_size = self.eps / 255.0
        targeted = self.targeted
        image_resizes = self.params.image_resizes
        image_width = self.params.image_width
        n_iters = self.params.n_iters
        interpol_dim = self.params.interpol_dim  # Not sure why this thing is different
        ODI_num_steps = 2

        if not isinstance(self.aux_models, dict):
            raise ValueError("Expected a dictionary of auxiliary models, since we will be working with an ensemble")
        # temporarily set these values for testing based on their original tf implementation
        x_min_val, x_max_val = 0, 1.0
        n_model_ensemble = len(self.aux_models)
        alpha = eps / n_iters

        adv = x_orig.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)
        sum_time = 0

        randVector_=0
        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            randVector_ += ch.FloatTensor(* model.forward(adv).shape).uniform_(-1., 1.).to('cuda')
        randVector_ = randVector_ / n_model_ensemble

        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            model.set_eval()  # Make sure model is in eval model
            model.zero_grad()  # Make sure no leftover gradients

        i = 0
        while self.optimization_loop_condition_satisfied(i, sum_time, n_iters + ODI_num_steps):
            if adv.grad is not None:
                adv.grad.zero_()
            start_time = time.time()

            if i == 0:
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)

            output = 0
            opt = optim.SGD([adv], lr=1e-3)
            opt.zero_grad()

            with ch.enable_grad():
                if i < ODI_num_steps:
                    loss=0
                    for model_name in self.aux_models:
                        model = self.aux_models[model_name]
                        loss += (model.forward(adv) * randVector_).sum()
                    loss = loss / n_model_ensemble
                else:
                    for model_name in self.aux_models:
                        model = self.aux_models[model_name]
                        output += model.forward(adv) / n_model_ensemble
                    output_clone = output.clone()
                    loss = self.criterion(output_clone, y_target)
            
            if self.config.track_local_metrics:
                with ch.no_grad():
                    current_local_loss = loss.item()
                    if i < ODI_num_steps:
                        current_local_asr = 0
                    else:
                        if targeted:
                            current_local_asr = ch.count_nonzero(ch.max(output_clone, 1).indices == y_target)
                        else:
                            current_local_asr = ch.count_nonzero(ch.max(output_clone, 1).indices != y_target)
                        current_local_asr = float(current_local_asr / len(y_target)) * 100

            loss.backward()
            if i < ODI_num_steps:
                eta = ODI_step_size * adv.grad.data.sign()
            else:
                eta = alpha * adv.grad.data.sign()
            if targeted == True:
                adv = adv - eta
            else:
                adv = adv + eta
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad=True)

            end_time = time.time()
            sum_time += end_time - start_time
            # outputs the transferability
            self.model.set_eval()  # Make sure model is in eval model
            self.model.zero_grad()  # Make sure no leftover gradients
            target_model_output = self.model.forward(adv)
            target_model_prediction = ch.max(target_model_output, 1).indices
            batch_size = len(y_target)
            if targeted:
                num_transfered = ch.count_nonzero(target_model_prediction == y_target)
            else:
                num_transfered = ch.count_nonzero(target_model_prediction != y_target)
            # print(num_transfered)
            transferability = float(num_transfered / batch_size) * 100

            with open(experiment_file_name, 'a') as f:
                f.write("iteration: %s" % (str(i)))
                f.write('\n')
                f.write("time: %s" % (str(sum_time)))
                f.write('\n')
                f.write("ASR: %s" % (str(transferability)))
                f.write('\n')
                if self.config.track_local_metrics:
                    f.write("local ASR: %s" % (str(current_local_asr)))
                    f.write('\n')
                    f.write("local loss: %s" % (str(current_local_loss)))
                    f.write('\n')

            del output, target_model_output, target_model_prediction
            ch.cuda.empty_cache()
            del loss
            gc.collect()  # Explicitly call the garbage collector

            i += 1

        stop_queries = 1

        # # outputs the transferability
        # self.model.set_eval()  # Make sure model is in eval model
        # self.model.zero_grad()  # Make sure no leftover gradients
        # target_model_output = self.model.forward(adv)
        # target_model_prediction = ch.max(target_model_output, 1).indices
        # batch_size = len(y_target)
        # if targeted:
        #     num_transfered = ch.count_nonzero(target_model_prediction == y_target)
        # else:
        #     num_transfered = ch.count_nonzero(target_model_prediction != y_target)
        # transferability = float(num_transfered / batch_size) * 100
        # # print("The transferbility of IFGSM is %s %%" % str(transferability))
        # self.logger.add_result(n_iters, {
        #     "transferability": str(transferability),
        # })
        return adv.detach(), stop_queries
