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


class NES_square(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = NESConfig(**self.params)
        self.x_final = None
        self.queries = 1
####    TO DO: change the loss function
        self.criterion = get_loss_fn("scel")
        self.norm = None
        self.k = 1


    def robust_in_top_k(self, target_class, proposed_adv):
        with ch.no_grad():
            proposed_adv = proposed_adv.type(ch.float)
            logits = self.model.forward(proposed_adv)
            vals, inds = ch.topk(logits, k=self.k)
        if target_class in inds:
            return True
        return False

    def attack(self, x_orig, x_adv, y_label, x_target, y_target):
        """
            Attack the original image using combination of transfer methods and return adversarial example
            (x, y_label): original image
        """
        eps = self.eps / 255.0
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0
        momentum = 0.9
        samples_per_draw = 100
        is_preturbed = False
        sigma = 1e-3
        batch_size = 10
        max_queries = 100000
        min_lr = 1e-3
        max_lr = 1e-2
        plateau_length = 20
        plateau_drop = 2.0
        last_ls = []
        g = 0
        num_queries = 0
        ret_adv=x_orig

        adv_thresh = 0.2
        # adv_thresh = 1
        # temp_eps=0.1
        delta_epsilon=0.5
        conservative=2


        for idx in range(len(x_orig)):
            temp_eps = 1

            print("###################===================####################")
            print(idx)

            stop_queries = 0
            x_image, initial_adv, target_label = x_orig[idx].unsqueeze(0),x_target[idx].unsqueeze(0), y_target[idx].int()
            lower = clip_by_tensor(x_image - temp_eps, x_min_val, x_max_val)
            upper = clip_by_tensor(x_image + temp_eps, x_min_val, x_max_val)
            adv = clip_by_tensor(initial_adv, lower, upper)
            self.model.set_eval()  # Make sure model is in eval model
            self.model.zero_grad()  # Make sure no leftover gradients
            print(f"Image {idx:d}   Target label: {target_label:d}")
            iter = 0
            success_flag=0
            transfer_flag=0
            while num_queries+1 < max_queries:
                print("i------------" + str(iter))
                iter += 1
                with ch.no_grad():
                    adv = adv.to(device='cuda', dtype=ch.float)
                    target_model_output = self.model.forward(adv)
                    target_model_prediction = ch.max(target_model_output, 1).indices
                    num_queries+=1
                    stop_queries+=1
                if targeted and target_model_prediction.item() == target_label.item() and (temp_eps <= eps):
                    if iter==1:
                        transfer_flag=1
                        num_transfer+=1
                    success_flag=1
                    num_success+=1
                    print("The image has been attacked! The attack used " + str(stop_queries) + " queries.")
                    ret_adv.append(adv)
                    break
                if stop_queries + 1 > max_queries:
                    print("Out of queries!")
                    break
                num_queries+=1
                stop_queries+=1
                print("Current label: " + str(target_model_prediction.item()) + "   loss: " + str(l.item())+"   eps: "+ str(temp_eps))


###             TO DO: get the loss value(object function)
                l= self.objection_function()
                print(loss)
                perturbation=self.calculate_perturbation()


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
                prop_de = 0.0

                adv_thresh=None#need to modify
                if l < adv_thresh and temp_eps > eps:
                    prop_de = delta_epsilon

                while current_lr >= min_lr:
                    # PARTIAL INFORMATION ONLY
                    proposed_epsilon = max(temp_eps - prop_de, eps)
                    lower = clip_by_tensor(x_image - proposed_epsilon, x_min_val, x_max_val)
                    upper = clip_by_tensor(x_image + proposed_epsilon, x_min_val, x_max_val)
                    # GENERAL LINE SEARCH
                    proposed_adv = adv.cpu() - targeted * current_lr * perturbation
                    proposed_adv = clip_by_tensor(proposed_adv.cuda(), lower, upper)
                    num_queries += 1
                    if self.robust_in_top_k(target_label, proposed_adv):
                        if prop_de > 0:
                            delta_epsilon = max(prop_de, 0.1)
                            last_ls = []
                        adv = proposed_adv
                        temp_eps = max(temp_eps - prop_de / conservative, eps)
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
                        print("[log] backtracking eps to %3f" % (temp_eps - prop_de,))

            ret_adv[idx]=adv
            self.logger.add_result(int(target_label.detach()), {
                "query": int(stop_queries),
                "transfer_flag": int(transfer_flag),
                "attack_flag": int(success_flag)
            })

        self.logger.add_result("Final Result", {
                        "success": int(num_success),
                        "image_avai": int(len(x_orig)-num_transfer),
                        "average query": int(num_queries/len(x_orig)),
                        "target model": str(self.model)
        })
        return ret_adv.detach(), num_queries
