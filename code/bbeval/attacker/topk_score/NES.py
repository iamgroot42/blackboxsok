import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_input import clip_by_tensor

np.set_printoptions(precision=5, suppress=True)


class NES(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = TransferredAttackConfig(**self.params)
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
            print(eval_points.shape)
            loss_val, noise = self.partial_info_loss(eval_points, noise, y_target, batch_size, labels)
            losses_tiled = ch.tile(ch.reshape(ch.tensor(loss_val), (-1, 1, 1, 1)),adv[0].shape)
            temp=ch.tensor(losses_tiled.cpu().detach().numpy() * noise)
            grad_val = ch.mean(temp, axis=0) / sigma
            losses.append(loss_val)
            grads.append(grad_val)
        losses=ch.cat(losses, dim=0)
        grads=ch.cat(grads, dim=0)
        return ch.mean(losses, dim=0), ch.mean(grads, dim=0)

    def partial_info_loss(self, eval_points, noise, y_target, batch_size, labels):
        with ch.no_grad():
            eval_points = eval_points.to(device='cuda', dtype=ch.float)
            logits = self.model.forward(eval_points)
        losses = self.criterion(logits=logits, labels=labels)
        print(losses)
        vals, inds = ch.topk(logits, k=self.k)
        good_inds = []
        # inds is batch_size x k
        for i in range(batch_size):
            for j in range(self.k):
                if inds[i][j] == y_target[i]:
                    good_inds.append(i)
                    continue
        # print(good_inds)
        losses = losses[good_inds]
        # print(losses)
        # s=ch.sum(losses)
        # print("SUM")
        # print(s)
        noise = noise[good_inds]
        return losses, noise

    def robust_in_top_k(self, target_class, proposed_adv):
        with ch.no_grad():
            proposed_adv = proposed_adv.type(ch.float)
            eval_logits_ = self.model.forward(proposed_adv)
            vals, inds = ch.topk(eval_logits_, k=self.k)
        print(inds)
        for i in range(len(inds)):
            if target_class[i] not in inds[i]:
                # print("false")
                # print(str(i)+str(target_class[i]))
                return False
        return True

    def attack(self, x_orig, x_adv, y_label, x_target, y_target):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        print(y_target)
        goal_epsilon = self.eps / 255.0
        eps_start = 1.0
        epsilon = eps_start
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0
        momentum = 0.9
        plateau_length = 20
        batch_size = 10
        min_lr = 1e-3
        adv_thresh = 0.5
        samples_per_draw = 10
        max_lr = 1e-2
        conservative = 2
        plateau_drop = 2.0
        stop_queries = 0
        g = 0
        last_ls = []
        num_queries=0
        delta_epsilon = 0.5
        prop_de=0.2

        labels = ch.zeros(batch_size, 1000)
        for row in range(batch_size):
            labels[row][y_target[row]] = 1

        adv = ch.clip(x_target, x_orig-epsilon, x_orig+epsilon)
        adv = adv.cuda()

        self.model.set_eval()  # Make sure model is in eval model
        self.model.zero_grad()  # Make sure no leftover gradients

        for i in range(self.n_iters):
            print("i------------"+str(i))
            stop_queries += 1
            #    check transferred iamges
            with ch.no_grad():
                adv = adv.to(device='cuda', dtype=ch.float)
                temp_adv = ch.clip(adv, x_orig-goal_epsilon, x_orig+goal_epsilon)
                target_model_output = self.model.forward(temp_adv)
                target_model_prediction = ch.max(target_model_output, 1).indices
            if targeted and ch.equal(target_model_prediction, y_target):
                raise ValueError("Yes sirrrrr")
                print('[log] early stopping at iteration %d' % stop_queries)
                return adv.detach(), stop_queries
            if (not targeted) and not (ch.equal(target_model_prediction, y_target)):
                print('[log] early stopping at iteration %d' % stop_queries)
                return adv.detach(), stop_queries

            prev_g = g
            l, g = self.get_grad(adv, samples_per_draw, batch_size, y_target, labels)
            # SIMPLE MOMENTUM
            g = momentum * prev_g + (1.0 - momentum) * g

            # # PLATEAU LR ANNEALING
            last_ls.append(l)
            last_ls = last_ls[-plateau_length:]
            if last_ls[-1] > last_ls[0] and len(last_ls) == plateau_length:
                if max_lr > min_lr:
                    print("[log] Annealing max_lr")
                    max_lr = max(max_lr / plateau_drop, min_lr)
                last_ls = []

            # SEARCH FOR LR AND EPSILON DECAY
            current_lr = max_lr

            prop_de=0.0
            if l < adv_thresh and epsilon > goal_epsilon:
                prop_de = delta_epsilon

            num=0
            while current_lr >= min_lr:
                print("num-----"+str(num))

                num+=1
                print(prop_de)
                # PARTIAL INFORMATION ONLY
                proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
                print(proposed_epsilon)
                lower = ch.clip(x_orig - proposed_epsilon, x_min_val, x_max_val)
                upper = ch.clip(x_orig + proposed_epsilon, x_min_val, x_max_val)
                # GENERAL LINE SEARCH
                proposed_adv = adv - (targeted * current_lr * ch.sign(g)).cuda()
                proposed_adv = ch.clip(proposed_adv, lower, upper)

                if self.robust_in_top_k(y_target, proposed_adv):
                    print("first")
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        last_ls = []

                    adv = proposed_adv
                    epsilon = max(epsilon - prop_de / conservative, goal_epsilon)
                    break
                elif current_lr >= min_lr * 2:
                    print("second")
                    current_lr = current_lr / 2
                else:
                    print("third")
                    prop_de = prop_de / 2
                    if prop_de == 0:
                        raise ValueError("Did not converge.")

                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = max_lr
                    print("[log] backtracking eps to %3f" % (epsilon - prop_de,))
            # proposed_adv = adv - (targeted * current_lr * ch.sign(g)).cuda()
            # while not self.robust_in_top_k(y_target, proposed_adv):
            #     if current_lr < min_lr * 2:
            #         epsilon = epsilon+prop_de
            #         prop_de=prop_de/2
            #         proposed_adv=x_adv
            #         break
            #     current_lr=current_lr/2
            #     proposed_adv = adv - (targeted * current_lr * ch.sign(g)).cuda()
            #     proposed_adv = ch.clip(proposed_adv, x_orig-epsilon, x_orig+epsilon)
            #
            # x_adv=proposed_adv
            # epsilon=epsilon-prop_de

        return adv.detach(), stop_queries

# 提高 adv_thresh
# 提高 conservative
# 提高k看看loss大小
# 查loss function