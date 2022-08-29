import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import StairCaseConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_input import clip_by_tensor

np.set_printoptions(precision=5, suppress=True)


class NES(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = StairCaseConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = get_loss_fn("scel")
        self.norm = None
        self.k = 5
        self.n_iters = self.params.n_iters

    def get_grad(self, adv, samples_per_draw, batch_size, y_target, labels):
        sigma = 1e-5
        num_batches = samples_per_draw // batch_size
        losses = []
        grads = []
        for _ in range(num_batches):
            noise_pos = np.random.normal(size=(batch_size // 2,) + adv[0].shape)
            noise = np.concatenate([noise_pos, -noise_pos], axis=0)
            eval_points = adv.cpu() + sigma * noise  # for scale
            losses, noise = self.partial_info_loss(eval_points, noise, y_target, batch_size, labels)
            print(losses)
            # loss_val = ch.tensor(np.asarray(loss_val))
            losses_tiled = ch.tile(ch.reshape(losses, (-1, 1)), ch.prod(adv[0].shape))
            losses_tiled = ch.reshape(losses_tiled, (batch_size,) + adv[0].shape)
            grad_val = ch.mean(losses_tiled * noise, axis=0) / sigma
            losses_val.append(loss_val)
            grads_val.append(grad_val)
        return torch.mean(losses), torch.mean(grads, dim=0)

    def partial_info_loss(self, eval_points, noise, y_target, batch_size, labels):
        eval_points = eval_points.to(device='cuda', dtype=ch.float)
        logits = self.model.forward(eval_points)
        losses = self.criterion(logits=logits, labels=labels)
        vals, inds = ch.topk(logits, k=self.k)
        good_inds = []
        # inds is batch_size x k
        for i in range(batch_size):
            for j in range(self.k):
                if inds[i][j] == y_target[i]:
                    good_inds.append(i)
                    continue
        losses = losses[good_inds]
        noise = noise[good_inds]
        return losses, noise

    def robust_in_top_k(self, target_class, proposed_adv, k_):
        eval_logits_ = self.model.forward(proposed_adv)
        if not target_class in eval_logits_.argsort()[-k_:][::-1]:
            return False
        return True

    def _attack(self, x_orig, x_adv=None, y_label=None, y_target=None):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        goal_epsilon = self.eps / 255.0
        eps_start = 0.5
        eps = eps_start
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0
        momentum = 0.9
        plateau_length = 20
        batch_size = 10
        min_lr = 1e-3
        adv_thresh = 0.2
        samples_per_draw = 10
        max_lr = 1e-2
        conservative = 2
        plateau_drop = 2.0
        stop_queries = 0
        g = 0

        # quite specific piece of code to staircase attack
        x_min = clip_by_tensor(x_orig - eps, x_min_val, x_max_val)
        x_max = clip_by_tensor(x_orig + eps, x_min_val, x_max_val)
        labels = ch.zeros(batch_size, 1000)
        for row in range(batch_size):
            labels[row][y_target[row]] = 1

        adv = clip_by_tensor(x_adv, x_min, x_max)
        adv = adv.cuda()
        self.model.set_eval()  # Make sure model is in eval model
        self.model.zero_grad()  # Make sure no leftover gradients

        for i in range(self.n_iters):
            stop_queries += 1
            #    check transferred iamges
            target_model_output = self.model.forward(x_orig)
            target_model_prediction = ch.max(target_model_output, 1).indices
            if targeted and ch.equal(target_model_prediction, y_target):
                print('[log] early stopping at iteration %d' % stop_queries)
                break
            if (not targeted) and not (ch.equal(target_model_prediction, y_target)):
                print('[log] early stopping at iteration %d' % stop_queries)
                break
            prev_g = g
            l, g = self.get_grad(adv, samples_per_draw, batch_size, y_target, labels)
            # SIMPLE MOMENTUM
            g = args.momentum * prev_g + (1.0 - momentum) * g

            # # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-plateau_length:]
            if last_ls[-1] > last_ls[0] and len(last_ls) == args.plateau_length:
                if max_lr > min_lr:
                    print("[log] Annealing max_lr")
                    max_lr = max(max_lr / plateau_drop, min_lr)
                last_ls = []

            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr
            proposed_adv = adv_x - targeted * current_lr * ch.sign(g)

            prop_de = 0.0
            if l < adv_thresh and epsilon > goal_epsilon:
                prop_de = delta_epsilon

            while current_lr >= min_lr:
                # PARTIAL INFORMATION ONLY
                proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
                lower = ch.clip(x - proposed_epsilon, 0, 1)
                upper = ch.clip(x + proposed_epsilon, 0, 1)
                # GENERAL LINE SEARCH
                proposed_adv = adv_x - targeted * current_lr * ch.sign(g)
                proposed_adv = ch.clip(proposed_adv, lower, upper)
                num_queries += 1

                if robust_in_top_k(target_class, proposed_adv, k):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        last_ls = []

                    adv_x = proposed_adv
                    epsilon = max(epsilon - prop_de / conservative, goal_epsilon)
                    break
                elif current_lr >= min_lr * 2:
                    current_lr = current_lr / 2
                else:
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        raise ValueError("Did not converge.")
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    print("[log] backtracking eps to %3f" % (epsilon - prop_de,))

            stop_queries += 1

        return adv.detach(), stop_queries
