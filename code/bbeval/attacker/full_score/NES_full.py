import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import NESConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_input import clip_by_tensor

np.set_printoptions(precision=5, suppress=True)


class NES_full(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = NESConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.criterion = get_loss_fn("scel")
        self.norm = None

    def get_grad(self, adv, samples_per_draw, batch_size, target_label):
        sigma = 1e-3
        num_batches = samples_per_draw // batch_size
        losses = []
        grads = []
        temp_label = ch.zeros(1000)
        temp_label[target_label]=1
        for _ in range(num_batches):
            noise_pos = np.random.normal(size=(batch_size // 2,) + adv[0].shape)
            noise = np.concatenate([noise_pos, -noise_pos], axis=0)
            eval_points = adv.cpu() + sigma * noise  # for scale
            eval_points = eval_points.to(device='cuda', dtype=ch.float)
            with ch.no_grad():
                logits = self.model.forward(eval_points)
                loss_val = self.criterion(logits, temp_label)
            losses_tiled = ch.tile(ch.reshape(loss_val, (-1, 1, 1, 1)), adv[0].shape)
            temp = ch.tensor(losses_tiled.cpu().detach().numpy() * noise)
            grad_val = ch.mean(temp, axis=0) / sigma
            losses.append(loss_val)
            grads.append(grad_val)
        losses = ch.cat(losses, dim=0)
        grads = ch.cat(grads, dim=0)
        return ch.mean(losses, dim=0), ch.mean(grads, dim=0)

    def attack(self, x_orig, x_adv, y_label, x_target, y_target):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        eps = self.eps / 255.0
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0
        momentum = 0.9
        samples_per_draw=50
        is_preturbed = False
        sigma=1e-3
        batch_size=10
        max_queries=100000
        min_lr = 5e-5
        max_lr = 1e-2
        plateau_length = 5
        plateau_drop = 2.0
        last_ls = []
        g = 0
        num_queries = 0
        ret_adv=[]

        for idx in range(len(x_orig)):

            print("###################===================####################")
            stop_queries = 0
            initial_img, target_label = x_orig[idx].unsqueeze(0), y_target[idx]
            lower = clip_by_tensor(initial_img - eps, x_min_val, x_max_val)
            upper = clip_by_tensor(initial_img + eps, x_min_val, x_max_val)
            adv = clip_by_tensor(initial_img, lower, upper)
            self.model.set_eval()  # Make sure model is in eval model
            self.model.zero_grad()  # Make sure no leftover gradients
            print(f"Image {idx:d}   Target label: {target_label:d}")
            iter=0
            while num_queries<max_queries:
                print("i------------" + str(iter))
                iter+=1
                with ch.no_grad():
                    adv = adv.to(device='cuda', dtype=ch.float)
                    target_model_output = self.model.forward(adv)
                    target_model_prediction = ch.max(target_model_output, 1).indices
                if is_preturbed:
                    num_queries+=sample_per_draw
                    stop_queries+=sample_per_draw
                else:
                    num_queries+=1
                    stop_queries+=1
                if (targeted and target_model_prediction.item()==target_label.item())\
                        or (not targeted and target_model_prediction.item()!= target_label.item()):
                    print("The image has been attacked! The attack used "+stop_queries+"queries.")
                    ret_adv.append(adv)
                    break
                    # print('[log] early stopping at iteration %d' % stop_queries)
                    # return adv.detach(), stop_queries
                prev_g = g
                l, g = self.get_grad(adv, samples_per_draw, batch_size, target_label)
                print("Current label: "+str(target_model_prediction.item())+"   loss: "+str(l.item()))
                # SIMPLE MOMENTUM
                g = momentum * prev_g + (1.0 - momentum) * g
                # PLATEAU LR ANNEALING
                last_ls.append(l)
                last_ls = last_ls[-plateau_length:]
                if last_ls[-1] > last_ls[0] and len(last_ls) == plateau_length:
                    if max_lr > min_lr:
                        print("[log] Annealing max_lr")
                        max_lr = max(max_lr / plateau_drop, min_lr)
                    last_ls = []
                # SEARCH FOR LR AND EPSILON DECAY
                current_lr = max_lr
                adv = adv - (targeted * current_lr * ch.sign(g)).cuda()
                adv = clip_by_tensor(adv, lower, upper)

        return ret_adv.detach(), num_queries

