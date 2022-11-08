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
        min_lr = 1e-5
        max_lr = 5
        plateau_length = 20
        plateau_drop = 2.0
        last_ls = []
        g = 0
        num_queries = 0
        ret_adv=x_orig
        num_transfer=0
        num_success=0
        adv_thresh = 0.2
        # adv_thresh = 1
        # temp_eps=0.1
        delta_epsilon=0.5
        conservative=2
        self.loss_type = "ce"
        self.succ =0


        for idx in range(len(x_orig)):
            temp_eps = 1
            idx+=10

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
            iter_sq=0
            decay_flag =True
            success_flag=0
            transfer_flag=0
            x_min, x_max = 0, 1 
            c, h, w = adv.shape[1:]
            n_features = c*h*w
            p_iter =0
            # probs = ch.softmax(logits, 1)
            target_label_l = ch.tensor([target_label]).cuda()
            target_label_l=target_label_l.to(device='cuda', dtype=ch.int64)

            while stop_queries+1 < max_queries:
                print("i------------" + str(iter))
                iter += 1
                p_iter +=1
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
                    break
                if stop_queries + 1 > max_queries:
                    print("Out of queries!")
                    break
                num_queries+=1
                stop_queries+=1

                x_best = ch.clip(ch.clip(adv,adv-(temp_eps),adv+temp_eps),0,1)
                logits = self.model.forward(x_best, detach=True)
                loss_min=l = get_loss_fn(self.loss_type, 'none')(logits, target_label_l)
                decay_flag =False
 
                ##Square Part
                
                while l>=loss_min :
                    
                    if decay_flag:
                        p_iter= 0
                        iter_sq=0
                        print("inital complete")
                        decay_flag =False
                        #print(init_delta,init_delta.shape)
                        #print(init_delta.sum())
                        x_best = ch.clip(ch.clip(adv,adv-(temp_eps),adv+temp_eps),0,1)
                        logits = self.model.forward(x_best, detach=True)
                        loss_min = get_loss_fn(self.loss_type, 'none')(logits, target_label_l)
                        num_queries+=1
                        stop_queries+=1

                    iter_sq +=1
                    deltas = x_best-adv
                    #print(deltas)
                    #print(deltas)
                    p = self.p_selection(0.05, iter_sq, max_queries)
                    #print(p)
                    s = int(round(np.sqrt(p * n_features / c)))
                    s = min(max(s, 1), h-1)
                    #print(s)
                    center_h = np.random.randint(0, h - s)
                    center_w = np.random.randint(0, w - s)

                    x_curr_window = adv[0,:,center_h:center_h+s, center_w:center_w+s]
                    #print(x_curr_window.shape)
                    x_adv_loc_curr_window = adv[0,:,center_h:center_h+s, center_w:center_w+s] 

                    # x_best_curr_window = x_best[0,:,center_h:center_h+s, center_w:center_w+s]
                    #print(x_best_curr_window.shape)
                    # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                    """
                    while ch.sum(ch.abs(ch.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], x_min, x_max) - x_best_curr_window) < 10**-7) == c*s*s:
                        deltas[i_img, :, center_h:center_h+s, center_w:center_w +
                            s] = self._workaround_choice([c, 1, 1], eps)
                    """
                    #print(ch.sum(ch.abs(ch.clip(ch.clip(x_adv_loc_curr_window + deltas[0,:, center_h:center_h+s, center_w:center_w+s],x_curr_window-temp_eps,x_curr_window+temp_eps), x_min, x_max) - x_adv_loc_curr_window) < 10**-7))
                    while ch.sum(ch.abs(ch.clip(ch.clip(x_adv_loc_curr_window + deltas[0,:, center_h:center_h+s, center_w:center_w+s],x_curr_window-temp_eps,x_curr_window+temp_eps), x_min, x_max) - x_adv_loc_curr_window) < 10**-7) == c*s*s:
                        deltas[0, :, center_h:center_h+s, center_w:center_w +s] = self._workaround_choice([c, 1, 1], 0.05)
                        #print("adfasff")
                        #print(deltas,deltas.shape)
                    #print(deltas)
                    x_new = ch.clip(ch.clip(adv+deltas,lower,upper),0,1)
                    logits = self.model.forward(x_new, detach=True)
                    #print(logits)
    ###             TO DO: get the loss value(object function)
                    l = get_loss_fn(self.loss_type, 'none')(logits, target_label_l)
                    #print("   loss: " + str(l.item())+"   eps: "+ str(temp_eps))
                    num_queries+=1
                    stop_queries+=1

                    if l<adv_thresh:
                        break
                    
                    #print(l)
                '''
                idx_improved = (l < loss_min)
                print(idx_improved,~idx_improved)
                loss_min = idx_improved * l + ~idx_improved * loss_min
                x_new= idx_improved * x_new + ~idx_improved * adv 
                perturbation = x_new-adv
                print(perturbation.sum())
                '''
                idx_improved = (l < loss_min)
                loss_min = idx_improved * l + ~idx_improved * loss_min
                x_best= idx_improved * x_new + ~idx_improved * adv 
                print("Current label: " + str(target_model_prediction.item()) + "   loss: " + str(l.item())+"   eps: "+ str(temp_eps)+"   query: "+ str(stop_queries))
                perturbation = x_best-adv



                # PLATEAU LR ANNEALING
                last_ls.append(loss_min)
                last_ls = last_ls[-plateau_length:]
                if last_ls[-1] > last_ls[0] and len(last_ls) == plateau_length:
                    if max_lr > min_lr:
                        print("[log] Annealing max_lr")
                        max_lr = max(max_lr / plateau_drop, min_lr)
                    last_ls = []
                # SEARCH FOR LR AND EPSILON DECAY
                current_lr = max_lr
                prop_de = 0.0

                adv_thresh = 1#need to modify
                if l < adv_thresh and temp_eps > eps:
                    prop_de = delta_epsilon

                while current_lr >= min_lr:
                    # PARTIAL INFORMATION ONLY
                    proposed_epsilon = max(temp_eps - prop_de, eps)
                    lower = clip_by_tensor(x_image - proposed_epsilon, x_min_val, x_max_val)
                    upper = clip_by_tensor(x_image + proposed_epsilon, x_min_val, x_max_val)
                    # GENERAL LINE SEARCH
                    proposed_adv = adv.cpu() + targeted * current_lr * perturbation.cpu()
                    #print(perturbation.sum())
                    proposed_adv = clip_by_tensor(proposed_adv.cuda(), lower, upper)
                    
                    num_queries += 1
                    stop_queries+=1
                    if self.robust_in_top_k(target_label, proposed_adv):
                        if prop_de > 0:
                            delta_epsilon = max(prop_de, 0.1)
                            last_ls = []
                        adv = proposed_adv
                        prev_eps=temp_eps
                        temp_eps = max(temp_eps - prop_de / conservative, eps)
                        if prev_eps!=prev_eps:
                            decay_flag=True
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


    

    def _workaround_choice(self, shape, eps=1.0):
        """
            Trick to generate numbers out of [-eps, eps]
            using numbers generated in [0, 1)
        """
        y = 2*(ch.rand(shape).cuda() - 0.5)
        y = ch.sign(y) * ch.abs(y) * eps
        return y

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init
  
        return p

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = ch.zeros([x, y]).cuda()
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
                    max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= ch.sqrt(ch.sum(delta ** 2, dim=(0, 1), keepdim=True))

        return delta

    def meta_pseudo_gaussian_pert(self, s):
        delta = ch.zeros([s, s]).cuda()
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s //
                    2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= ch.sqrt(ch.sum(delta ** 2, dim=(0, 1), keepdim=True))
            if ch.rand(1).item() > 0.5:
                delta = delta.T

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(
                s // 2, s // 2) * self._workaround_choice(1)
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(
                s - s // 2, s // 2) * self._workaround_choice(1)
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(
                s // 2, s - s // 2) * self._workaround_choice(1)
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(
                s - s // 2, s - s // 2) * self._workaround_choice(1)
            delta /= ch.sqrt(ch.sum(delta ** 2, dim=(0, 1), keepdim=True))

        return delta