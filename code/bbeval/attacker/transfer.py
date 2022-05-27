import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.attacker.transfer_methods.manipulate_gradient import torch_staircase_sign, project_noise
from bbeval.attacker.transfer_methods.manipulate_input import ensemble_input_diversity

np.set_printoptions(precision=5, suppress=True)


class Transfer(Attacker):
    def __init__(self, model: GenericModelWrapper, config: AttackerConfig):
        super().__init__(model, config)
        self.x_final = None
        self.queries = 1

    def attack(self, x, y, eps, **kwargs):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y): original image
        """
        models = kwargs.get('models')
        if not isinstance(models, dict):
            raise ValueError("Expected a dictionary of models, since we will be working with an ensemble")
        n_iters = kwargs.get('n_iters')
        n_ensemble = kwargs.get('n_ensemble')
        amplification = kwargs.get('amplification')
        rescaled_dim = kwargs.get('rescaled_dim')
        alpha = eps / n_iters
        alpha_beta = alpha * amplification
        gamma = alpha_beta

        # initializes the advesarial example
        # x.requires_grad = True
        adv = x.clone()
        adv = adv.cuda()
        adv.requires_grad = True
        amplification = 0.0
        pre_grad = ch.zeros(adv.shape).cuda()
        # quite specific piece of code to staircase attack
        # TODO: THe 0/1 below should be dataset-specific
        x_min = clip_by_tensor(x - eps, 0.0, 1.0)
        x_max = clip_by_tensor(x + eps, 0.0, 1.0)

        # use for loop to replace the original manual attack process
        """
        eff = models["eff"]
        dense = models['dense']
        res = models['res50']
        res101 = models['res101']
        # wide = models['wide']
        dense169 = models['dense169']
        vgg = models['vgg']
        # lpipsLoss = models["lpips"]

        res101.zero_grad()
        eff.zero_grad()
        dense.zero_grad()
        dense169.zero_grad()
        res.zero_grad()
        # wide.zero_grad(True)
        vgg.zero_grad()
        # lpipsLoss.zero_grad(True)
        """
        for model in models:
            model.zero_grad()

        # start the main attack process: ensemble input diveristy as a demo
        for i in range(n_iters):
            if i == 0:
                adv = F.conv2d(adv, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                adv = clip_by_tensor(adv, x_min, x_max)
                adv = V(adv, requires_grad = True)

            """
            output1 = 0
            
            output1 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            output1 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            output1 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            output1 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            output1 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            output1 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./6
            # output1 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 0), (256, 256), mode='bilinear')) * 1./7
            loss1 = F.cross_entropy(output1 * 1.5, gt, reduction="none")

            output3 = 0
            output3 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            output3 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            output3 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            output3 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            output3 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            output3 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 1), (256, 256), mode='bilinear')) * 1./6
            # output3 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./7
            loss3 = F.cross_entropy(output3 * 1.5, gt, reduction="none")

            output4 = 0
            output4 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            output4 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            output4 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            output4 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            output4 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            output4 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 2), (256, 256), mode='bilinear')) * 1./6
            # output4 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./7
            loss4 = F.cross_entropy(output4 * 1.5, gt, reduction="none")

            output5 = 0
            output5 += dense(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            output5 += res(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            output5 += res101(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            output5 += dense169(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            output5 += vgg(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            output5 += eff(F.interpolate(ensemble_input_diversity(adv + pre_grad, 3), (256, 256), mode='bilinear')) * 1./6
            # output5 += wide(F.interpolate(ensemble_input_diversity(adv + pre_grad, 4), (256, 256), mode='bilinear')) * 1./7
            loss5 = F.cross_entropy(output5 * 1.5, gt, reduction="none")

            loss = (loss1 + loss3 + loss4 + loss5) / 4.0
            """
            loss = 0
            for i in range(n_ensemble):
                output = 0
                for model in models:
                    output += model(F.interpolate(ensemble_input_diversity(adv + pre_grad, i), (rescaled_dim, rescaled_dim), mode='bilinear')) * 1./6
                loss += F.cross_entropy(output * 1.5, y, reduction="none")
            loss  = loss/n_ensemble
            loss.mean().backward()
            noise = adv.grad.data
            pre_grad = adv.grad.data
            noise = F.conv2d(noise, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)

            # MI-FGSM
            # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
            # noise = momentum * grad + noise
            # grad = noise

            # PI-FGSM
            amplification += alpha_beta * torch_staircase_sign(noise, 1.5625)
            cut_noise = clip_by_tensor(abs(amplification) - eps, 0.0, 10000.0) * ch.sign(amplification)
            projection = alpha * torch_staircase_sign(project_noise(cut_noise, stack_kern, kern_size), 1.5625)

            # staircase sign method (under review) can effectively boost the transferability of adversarial examples, and we will release our paper soon.
            pert = (alpha_beta * torch_staircase_sign(noise, 1.5625) + 0.5 * projection) * 0.75
            # adv = adv + pert * (1-mask) * 1.2 + pert * mask * 0.8
            adv = adv + pert
            # print(mask.max())
            # print(mask.min())
            # exit()
            # adv = adv + alpha * torch_staircase_sign(noise, 1.5625)
            adv = clip_by_tensor(adv, x_min, x_max)
            adv = V(adv, requires_grad = True)
        
        stop_queries = 1
        return adv.detach(), stop_queries
