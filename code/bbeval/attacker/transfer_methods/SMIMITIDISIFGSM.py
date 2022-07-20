import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import StairCaseConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_gradient import torch_staircase_sign, project_noise, gkern, \
    project_kern
from bbeval.attacker.transfer_methods._manipulate_input import ensemble_input_diversity, input_diversity, clip_by_tensor

import torchvision.models as models  # TODO: remove after test

np.set_printoptions(precision=5, suppress=True)


# https://arxiv.org/pdf/2203.13479.pdf
# n_iters=20
# num_transformations=12
# m=5

class SMIMITIDISIFGSM(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = StairCaseConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = get_loss_fn("ce")
        self.norm = None

    def transformation_function(self, x):
        img_size = x.shape[-1]
        img_resize = 270
        rnd = ch.randint(low=img_resize, high=img_size, size=(1,), dtype=ch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_size - rnd
        w_rem = img_size - rnd
        pad_top = ch.randint(low=0, high=h_rem.item(), size=(1,), dtype=ch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = ch.randint(low=0, high=w_rem.item(), size=(1,), dtype=ch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded

    def input_diversity(self, x,img_resize):
        diversity_prob = 0.5
        img_size = x.shape[-1]

        rnd = ch.randint(low=img_size, high=img_resize, size=(1,), dtype=ch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = ch.randint(low=0, high=h_rem.item(), size=(1,), dtype=ch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = ch.randint(low=0, high=w_rem.item(), size=(1,), dtype=ch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if ch.rand(1) < diversity_prob else x

    def _attack(self, x_orig, x_adv=None, y_label=None, y_target=None):
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
        decay = 1.0
        momentum = 0
        num_transformations = 12
        lamda = 1 / num_transformations
        Gradients = []
        m=3

        # initializes the advesarial example
        # x.requires_grad = True
        adv = x_orig.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        pre_grad = ch.zeros(adv.shape).cuda()
        # quite specific piece of code to staircase attack
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)

        kernel_size = 5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = ch.from_numpy(gaussian_kernel).cuda()

        for model_name in self.aux_models:
            model = self.aux_models[model_name]
            model.set_eval()  # Make sure model is in eval model
            model.zero_grad()  # Make sure no leftover gradients
            # logits_clean = model.forward(x_orig, detach=True)
            # corr_classified = ch.argmax(logits_clean, dim=1) == y_label
            # print('Clean accuracy of candidate samples: {:.2%}'.format(ch.mean(1. * corr_classified).item()))

        for i in range(n_iters):
            if i == 0:
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad=True)

            print("iteration "+str(i))
            grad = 0
            for t in range(num_transformations):
                print("transformation "+str(t))
                input = self.transformation_function(adv)
                grad_temp = 0
                for j in ch.arange(1):
                    print(j)
                    x_nes = input / ch.pow(2, j)
                    x_nes = V(x_nes, requires_grad=True)
                    output = 0
                    for model_name in self.aux_models:
                        model = self.aux_models[model_name]
                        output += model.forward(self.input_diversity(x_nes, image_resizes[0])) / n_model_ensemble

                    output_clone = output.clone()
                    loss = self.criterion(output_clone, y_target, targeted)
                    print(loss)
                    loss.backward()
                    grad_temp += x_nes.grad.data/m

                Gradients.append(grad_temp)


            for gradient in Gradients:
                grad += lamda * gradient

            grad = F.conv2d(grad, gaussian_kernel, stride=1, padding='same', groups=3)
            grad = momentum * decay + grad / ch.mean(ch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = grad

            if targeted == True:
                adv = adv - alpha * ch.sign(grad)
            else:
                adv = adv + alpha * ch.sign(grad)
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad=True)

        stop_queries = 1

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
        transferability = float(num_transfered / batch_size) * 100
        print("The transferbility of SMIMITIDISIFGSM is %s %%" % str(transferability))
        self.logger.add_result(n_iters, {
            "transferability": str(transferability),
        })
        return adv.detach(), stop_queries
