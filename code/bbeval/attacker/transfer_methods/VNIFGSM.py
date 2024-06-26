import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_input import  clip_by_tensor

import time
import gc

np.set_printoptions(precision=5, suppress=True)


class VNIFGSM(Attacker):
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
        targeted = self.targeted
        image_resizes = self.params.image_resizes
        image_width = self.params.image_width
        n_iters = self.params.n_iters
        interpol_dim = self.params.interpol_dim  # Not sure why this thing is different

        if not isinstance(self.aux_models, dict):
            raise ValueError("Expected a dictionary of auxiliary models, since we will be working with an ensemble")
        # temporarily set these values for testing based on their original tf implementation
        x_min_val, x_max_val = 0, 1.0
        image_width = 299
        image_resizes = [330]
        interpol_dim = 256

        n_model_ensemble = len(self.aux_models)
        n_input_ensemble = len(image_resizes)
        alpha = eps / n_iters
        # decay (float): momentum factor. (Default: 1.0)
        # N (int): the number of sampled examples in the neighborhood. (Default: 20)
        # beta (float): the upper bound of neighborhood. (Default: 3/2)
        decay = 1.0
        grad = 0
        v=0
        momentum = 0
        N = 20
        beta = 3 / 2

        # initializes the advesarial example
        # x.requires_grad = True
        adv = x_orig.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        pre_grad = ch.zeros(adv.shape).cuda()
        # quite specific piece of code to staircase attack
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)
        sum_time = 0

        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            model.set_eval()  # Make sure model is in eval model
            model.zero_grad()  # Make sure no leftover gradients

        i = 0
        while self.optimization_loop_condition_satisfied(i, sum_time, n_iters):
            if adv.grad is not None:
                adv.grad.zero_()
            start_time = time.time()
            if i == 0:
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)

            output = 0
            x_nes = adv + decay * alpha * momentum
            for model_name in self.aux_models:
                model = self.aux_models[model_name]
                output += model.forward(x_nes) / n_model_ensemble

            output_clone = output.clone()
            loss = self.criterion(output_clone, y_target)

            if self.config.track_local_metrics:
                with ch.no_grad():
                    current_local_loss = loss.item()
                    if targeted:
                        current_local_asr = ch.count_nonzero(ch.max(output_clone, 1).indices == y_target)
                    else:
                        current_local_asr = ch.count_nonzero(ch.max(output_clone, 1).indices != y_target)
                    current_local_asr = float(current_local_asr / len(y_target)) * 100

            # print(i)
            # print(loss)
            loss.backward()
            adv_grad = adv.grad.data
            grad = momentum * decay + (adv_grad+v) / ch.mean(ch.abs(adv_grad+v), dim=(1, 2, 3), keepdim=True)
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = 0
            for _ in range(N):
                neighbor_images = adv.detach() + ch.randn_like(x_orig).uniform_(-eps*beta, eps*beta)
                neighbor_images.requires_grad = True
                output = 0
                for model_name in self.aux_models:
                    model = self.aux_models[model_name]
                    output += model.forward(neighbor_images) / n_model_ensemble

                output_clone = output.clone()
                # Calculate loss
                if targeted:
                    cost = -self.criterion(output_clone, y_target)
                else:
                    cost = self.criterion(output_clone, y_target)
                cost.backward()
                GV_grad += neighbor_images.grad.data
            # obtaining the gradient variance
            v = GV_grad / N - adv_grad

            if targeted:
                adv = adv - alpha * ch.sign(grad)
            else:
                adv = adv + alpha * ch.sign(grad)
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

            del output, output_clone, target_model_output, target_model_prediction
            ch.cuda.empty_cache()
            del loss
            gc.collect()  # Explicitly call the garbage collector

            i += 1

        stop_queries = 1

        # outputs the transferability
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
        # print("The transferbility of VNIFGSM is %s %%" % str(transferability))
        # self.logger.add_result(n_iters, {
        #     "transferability": str(transferability),
        # })
        return adv.detach(), stop_queries
