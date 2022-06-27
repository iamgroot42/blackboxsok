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


class IFGSM(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = StairCaseConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = None
        self.norm = None


    def _attack(self, x_orig, x_adv=None, y_label=None, y_target=None):
        # for model_name in self.aux_models:
        #     model = self.aux_models[model_name]
        #     print(model.forward(x_orig))
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        # pytorch version: n_iter=40, eps=20/255
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

        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            model.set_eval()  # Make sure model is in eval model
            model.zero_grad()  # Make sure no leftover gradients

        for i in range(n_iters):
            if i == 0:
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)
            loss = 0
            for image_resize in image_resizes:
                output = 0
                for model_name in self.aux_models:
                    model = self.aux_models[model_name]
                    output += model.forward(adv) / n_model_ensemble
                    # output += model.forward(input_diversity(adv + pre_grad, image_width, image_resize)) * 1./n_model_ensemble
                    output1=output.clone()
                    if targeted:
                        loss += F.cross_entropy(output1 , y_target,
                                                reduction="none")
                    else:
                        loss -= F.cross_entropy(output1 , y_target,
                                                reduction="none")
            if adv.grad is not None:
                adv.grad.data.fill_(0)
            loss = loss / n_input_ensemble
            loss.mean().backward()

            adv = adv - alpha*adv.grad.data
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

        print("The transferbility of IFGSM is %s %%" % str(transferability))
        self.logger.add_result(n_iters, {
            "transferability": str(transferability),
        })
        return adv.detach(), stop_queries