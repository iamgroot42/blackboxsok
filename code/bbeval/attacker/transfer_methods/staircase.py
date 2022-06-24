import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import StairCaseConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.attacker.transfer_methods._manipulate_gradient import torch_staircase_sign, project_noise, gkern, \
    project_kern
from bbeval.attacker.transfer_methods._manipulate_input import ensemble_input_diversity, input_diversity, clip_by_tensor

import torchvision.models as models  # TODO: remove after test

np.set_printoptions(precision=5, suppress=True)


# untarget: https://github.com/qilong-zhang/Staircase-sign-method/blob/main/attack_iter_SSM_EAT.py
# target: https://github.com/qilong-zhang/CVPR2021-Competition-Unrestricted-Adversarial-Attacks-on-ImageNet/blob/main/run.py

class Staircase(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = StairCaseConfig(**self.params)
        self.x_final = None
        self.queries = 1

    def _attack(self, x_orig, x_adv=None, y_label=None, y_target=None):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        eps = self.eps / 255.0
        targeted = self.targeted
        image_resizes = self.params.image_resizes
        image_width = self.params.image_width
        prob = self.params.prob
        amplification = self.params.amplification
        n_iters = self.params.n_iters
        interpol_dim = self.params.interpol_dim  # Not sure why this thing is different

        if not isinstance(self.aux_models, dict):
            raise ValueError("Expected a dictionary of auxiliary models, since we will be working with an ensemble")
        # temporarily set these values for testing based on their original tf implementation
        amplification = 10  # amplification_factor: 10.0 for tensorflow implementation
        # TODO: Should be [-1, 1] for data clip?
        x_min_val, x_max_val = 0, 1.0
        image_width = 299
        image_resizes = [330]
        interpol_dim = 256
        prob = 0.7

        n_model_ensemble = len(self.aux_models)
        n_input_ensemble = len(image_resizes)
        alpha = self.eps / n_iters
        alpha_beta = alpha * amplification
        gamma = alpha_beta * 0.8

        # initializes the advesarial example
        # x.requires_grad = True
        adv = x_orig.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        # amplification = 0.0 # TODO: check what this is actually doing
        pre_grad = ch.zeros(adv.shape).cuda()
        # quite specific piece of code to staircase attack
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)

        # Create Gaussian kernel
        kernel_size = 5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = ch.from_numpy(gaussian_kernel).cuda()

        stack_kern, kern_size = project_kern(3)

        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            model.set_eval()  # Make sure model is in eval model
            model.zero_grad()  # Make sure no leftover gradients

        if targeted == False:
            # start the main attack process: ensemble input diveristy as a demo
            for i in range(n_iters):
                if i == 0:
                    adv = F.conv2d(adv, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                    adv = clip_by_tensor(adv, x_min, x_max)
                    adv = V(adv, requires_grad=True)
                loss = 0
                for image_resize in image_resizes:
                    output = 0
                    for model_name in self.aux_models:
                        model = self.aux_models[model_name]
                        output += model.forward(F.interpolate(
                            ensemble_input_diversity(adv + pre_grad, list(self.aux_models.keys()).index(model_name),
                                                     image_resize), (interpol_dim, interpol_dim),
                            mode='bilinear')) * 1. / n_model_ensemble
                        # output += model.forward(input_diversity(adv + pre_grad, image_width, image_resize)) * 1./n_model_ensemble
                        loss += F.cross_entropy(output * 1.5, y_target,
                                                reduction="none")  # TODO: this one should be amplification factor? cannot verify in the original implementation
                loss = loss / n_input_ensemble
                loss.mean().backward()
                noise = adv.grad.data
                pre_grad = adv.grad.data
                noise = F.conv2d(noise, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)

                # MI-FGSM
                # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
                # noise = momentum * grad + noise
                # grad = noise

                # PI-FGSM
                amplification += alpha_beta * torch_staircase_sign(noise, 1.5625)
                cut_noise = clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * ch.sign(amplification)
                projection = gamma * torch_staircase_sign(project_noise(cut_noise, stack_kern, kern_size), 1.5625)

                # staircase sign method (under review) can effectively boost the transferability of adversarial examples, and we will release our paper soon.
                adv = adv - alpha_beta * torch_staircase_sign(noise, 1.5625) - projection
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)
            stop_queries = 1

            # outputs the transferability
            # target_model_output=self.model.forward(x)
            target_model_output = self.model.forward(adv)
            target_model_prediction = ch.max(target_model_output, 1).indices
            batch_size = len(y_target)
            # print(target_model_prediction==y)
            num_transfered = ch.count_nonzero(target_model_prediction == y_target)
            transferability = float(num_transfered / batch_size) * 100

        else:
            # start the main attack process: ensemble input diveristy as a demo
            for i in range(n_iters):
                if i == 0:
                    adv = F.conv2d(adv, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                    adv = clip_by_tensor(adv, x_min, x_max)
                    adv = V(adv, requires_grad=True)
                loss = 0
                for image_resize in image_resizes:
                    output = 0
                    for model_name in self.aux_models:
                        model = self.aux_models[model_name]
                        output += model.forward(F.interpolate(
                            ensemble_input_diversity(adv + pre_grad, list(self.aux_models.keys()).index(model_name),
                                                     image_resize), (interpol_dim, interpol_dim),
                            mode='bilinear')) * 1. / n_model_ensemble
                        # output += model.forward(input_diversity(adv + pre_grad, image_width, image_resize)) * 1./n_model_ensemble
                        loss += F.cross_entropy(output, y_target,
                                                reduction="none")  # TODO: this one should be amplification factor? cannot verify in the original implementation
                loss = loss / n_input_ensemble
                loss.mean().backward()
                noise = adv.grad.data
                pre_grad = adv.grad.data
                noise = F.conv2d(noise, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)

                # MI-FGSM
                # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
                # noise = momentum * grad + noise
                # grad = noise

                # PI-FGSM
                amplification += alpha_beta * torch_staircase_sign(noise, 1.5625)
                cut_noise = clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * ch.sign(amplification)
                projection = gamma * torch_staircase_sign(project_noise(cut_noise, stack_kern, kern_size), 1.5625)

                # staircase sign method (under review) can effectively boost the transferability of adversarial examples, and we will release our paper soon.
                adv = adv - alpha_beta * torch_staircase_sign(noise, 1.5625) - projection
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)
            stop_queries = 1

            # outputs the transferability
            # target_model_output=self.model.forward(x)
            target_model_output = self.model.forward(adv)
            target_model_prediction = ch.max(target_model_output, 1).indices
            batch_size = len(y_target)
            # print(target_model_prediction==y)
            num_transfered = ch.count_nonzero(target_model_prediction == y_target)
            transferability = float(num_transfered / batch_size) * 100

        print("The transferbility of Staircase is %s %%" % str(transferability))
        self.logger.add_result(n_iters, {
            "transferability": str(transferability),
        })
        return adv.detach(), stop_queries
