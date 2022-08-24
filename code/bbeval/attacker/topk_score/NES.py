import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import StairCaseConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn

import torchvision.models as models  # TODO: remove after test

np.set_printoptions(precision=5, suppress=True)


class NES(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = StairCaseConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = get_loss_fn("ce")
        self.norm = None

    def _attack(self, x_orig, x_adv=None, y_label=None, y_target=None):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        eps_bound = self.eps / 255.0
        eps_start = 0.5
        eps = eps_start
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0

        # quite specific piece of code to staircase attack
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)

        x_target
        adv = clip_by_tensor(x_target, x_min, x_max)
        adv = adv.cuda()
        stop_queries = 1

        while eps_start > eps_bound or top != y_target:
            g = NESEstgrad()
            lr = lr_max
            adv_hat = adv - is_targeted * lr * ch.sign(g)
            while y_target not in topk:
                if lr < lr_min:
                    eps = eps + decay
                    decay = decay / 2
                    adv_hat = adv
                    break
                lr = lr / 2
                adv_hat = clip_by_tensor(adv - lr * g, x_min, x_max)
            adv = adv_hat
            eps = eps - decay

        stop_queries += 1

        return adv.detach(), stop_queries
