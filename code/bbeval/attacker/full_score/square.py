import numpy as np
import torch as ch
from bbeval.models.core import GenericModelWrapper
from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig, SquareAttackConfig, ExperimentConfig
from bbeval.loss import get_loss_fn

from tqdm import tqdm

import time
np.set_printoptions(precision=5, suppress=True)


class Square_Attack(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig, experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = SquareAttackConfig(**self.params)
        
        self.norm_type = np.inf
        self.loss_type = "ce"
        self.succ =0

    
    def _workaround_choice(self, shape, eps=1.0):
        """
            Trick to generate numbers out of [-eps, eps]
            using numbers generated in [0, 1)
        """
        y = 2*(ch.rand(shape).cuda() - 0.5)
        y = ch.sign(y) * ch.abs(y) * eps
        return y

    def attack(self, x_orig, x_adv, y_label,x_target=None,y_target=None):
        x_adv_loc = x_adv
        #print(x_orig.shape[0])
        batch_num = int(x_orig.shape[0]/10)
        suc_num =0
        for batch in range(batch_num):
            x_orig_ = x_orig[batch*10: batch*10+10]
            x_adv_loc_ = x_adv_loc[batch*10: batch*10+10]
            y_label_ = y_label[batch*10: batch*10+10]
            y_target_ = y_target[batch*10: batch*10+10]
            # important to set the model to evaluation mode for square attack
            self.model.set_eval()
            # TODO: Add support for x_adv
            #n_iters = self.params.n_iters
            n_iters = 10000
            p_init = self.params.p_init
            # Don't need gradients for the attack, detach x if it has gradient collection on
            x_orig_, x_adv_loc_ = x_orig_.detach(), x_adv_loc_.detach()

            # distinguish between correctly classified and misclassified samples
            logits_clean = self.model.forward(x_orig_, detach=True)
            
            corr_classified = ch.argmax(logits_clean,dim=1) == y_label_
            #print(corr_classified)
            print('Clean accuracy of candidate samples: {:.2%}'.format(ch.mean(1.* corr_classified).item()))

            # the logic is untargeted attack does not have target labels, so y_target = y_label
            # y_label is mainly used for the purpose of measuring clean model performance.

            # # Use appropriate labels for the attack
            # if self.targeted:
            #     y_use = y_target
            # else:
            #     y_use = y_label
            #print(x_orig_.shape)
            
            query_count = 0
            #x_final = []
            if self.norm_type == 2:
                num_queries,x_perturbed  = self.square_attack_l2(
                    x_orig_, x_adv_loc_, y_target_, corr_classified, n_iters, p_init)
                print(num_queries.shape[0])
                for idx in range(10):
                    #print(x_perturbed[idx].unsqueeze(0).shape)
                    label = ch.argmax(self.model.forward(x_orig_[idx].unsqueeze(0)))
                    query_c = int(num_queries[idx])
                    #print(label,x_perturbed[idx])
                    transfer_flag = False
                    query_count+=query_c
                    self.logger.add_result(int(label.detach()), {
                            "query": int(query_c),
                            "transfer_flag": int(transfer_flag),
                            "attack_flag": int(query_c<n_iters)
                        })
                self.logger.add_result("Final Result", {
                        "success": int(self.succ),
                        "image_avai": int(len(x_orig)),
                        "average query": int((query_count/int(len(x_orig))))
                    })  
            elif self.norm_type ==  np.inf:
                if not self.targeted:
                    num_queries,x_perturbed,idx_t,loss = self.square_attack_linf(
                        x_orig_, x_adv_loc_, y_label_, corr_classified, n_iters, p_init)
                else:

                    num_queries,x_perturbed,idx_t,loss = self.square_attack_linf(
                        x_orig_, x_adv_loc_, y_target_, corr_classified, n_iters, p_init)
                print(num_queries)
                print(idx_t)
                #x_final.append(x_perturbed)
                for idx in range(10):
                    #print(x_perturbed[idx].unsqueeze(0).shape)
                    label = ch.argmax(self.model.forward(x_orig_[idx].unsqueeze(0)))
                    query_c = int(num_queries[idx])
                    loss_ = float(loss[idx])
                    #print(label,num_queries[idx])
                    transfer_flag = False
                    attack_flag=0
                    if not idx_t[idx] :
                        suc_num+=1
                        attack_flag =1

                    query_count+=query_c
                    self.logger.add_result(int(label.detach()), {
                            "query": int(query_c),
                            "transfer_flag": int(transfer_flag),
                            "attack_flag": int(attack_flag),
                            "loss_at_succ": float(loss_)
                        })
            else:
                raise NotImplementedError("Unsupported Norm Type!")
        self.logger.add_result("Final Result", {
        "success": int(suc_num),
        "image_avai": int(len(x_orig)),
        "average query": float((query_count/int(len(x_orig))))
        })          
        return x_perturbed, num_queries

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

    def square_attack_l2(self, x, x_adv_loc, y, corr_classified, n_iters, p_init):
        """ The L2 square attack """
        if self.seed is not None:
            ch.random.seed(self.seed)
        
        eps = self.eps/255
        x_min, x_max = 0, 1
        c, h, w = x.shape[1:]
        n_features = c * h * w
        n_ex_total = x.shape[0]
        # Ensure the passed attack seeds are always correctly classified!
        x, x_adv_loc, y = x[corr_classified], x_adv_loc[corr_classified], y[corr_classified]

        ### initialization
        delta_init = ch.zeros_like(x).cuda()
        s = h // 5
        self.logger.log('Initial square side={} for bumps'.format(s))
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for _ in range(h // s):
            center_w = sp_init + 0
            for _ in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * self._workaround_choice([x.shape[0], c, 1, 1])
                center_w += s
            center_h += s

        # x_best = ch.clip(x + delta_init / ch.sqrt(ch.sum(delta_init **
        #                  2, dim=(1, 2, 3), keepdim=True)) * eps, x_min, x_max)
        # check whether the operation below is optimal for l2 attacks
        x_best = x_adv_loc + delta_init / ch.sqrt(ch.sum(delta_init **
                         2, dim=(1, 2, 3), keepdim=True)) * eps
        diff_to_x = x_best - x
        x_best = x + diff_to_x / ch.sqrt(ch.sum(diff_to_x **
                         2, dim=(1, 2, 3), keepdim=True)) * eps
        x_best = ch.clip(x_best, x_min, x_max)

        logits = self.model.forward(x_best, detach=True)
        # probs = ch.softmax(logits, 1)
        loss_min = get_loss_fn(self.loss_type, 'none')(logits, y)
        margin_min = get_loss_fn('margin', 'none')(logits, y)
        # ones because we have already used 1 query
        n_queries = ch.ones(x.shape[0]).cuda()

        time_start = time.time()
        s_init = int(np.sqrt(p_init * n_features / c))
        for i_iter in range(n_iters):
            idx_to_fool = (margin_min > 0.0)

            x_curr, x_adv_loc_curr, x_best_curr = x[idx_to_fool], x_adv_loc[idx_to_fool], x_best[idx_to_fool]
            y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]
            # further modified to work with x_adv_loc
            # delta_curr = x_best_curr - x_curr
            delta_curr = x_best_curr - x_adv_loc_curr

            p = self.p_selection(p_init, i_iter, n_iters)
            s = max(int(round(np.sqrt(p * n_features / c))), 3)

            if s % 2 == 0:
                s += 1

            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = ch.zeros_like(x_curr).cuda()
            new_deltas_mask[:, :, center_h:center_h +
                            s, center_w:center_w + s] = 1.0

            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = ch.zeros_like(x_curr).cuda()
            new_deltas_mask_2[:, :, center_h_2:center_h_2 +
                              s2, center_w_2:center_w_2 + s2] = 1.0
            norms_window_2 = ch.sqrt(
                ch.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, dim=(-2, -1),
                       keepdim=True))

            ### compute total norm available
            """
            curr_norms_window = ch.sqrt(
                ch.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, dim=(2, 3), keepdim=True))
            curr_norms_image = ch.sqrt(
                ch.sum((x_best_curr - x_curr) ** 2, dim=(1, 2, 3), keepdim=True))
            mask_2 = ch.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = ch.sqrt(
                ch.sum((delta_curr * mask_2) ** 2, dim=(2, 3), keepdim=True))
            """
            # further modified to work with x_adv
            curr_norms_window = ch.sqrt(
                ch.sum(((x_best_curr - x_adv_loc_curr) * new_deltas_mask) ** 2, dim=(2, 3), keepdim=True))
            curr_norms_image = ch.sqrt(
                ch.sum((x_best_curr - x_adv_loc_curr) ** 2, dim=(1, 2, 3), keepdim=True))
            mask_2 = ch.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = ch.sqrt(
                ch.sum((delta_curr * mask_2) ** 2, dim=(2, 3), keepdim=True))


            ### create the updates
            new_deltas = ch.ones([x_curr.shape[0], c, s, s]).cuda()
            new_deltas = new_deltas * \
                self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= self._workaround_choice([x_curr.shape[0], c, 1, 1])
            old_deltas = delta_curr[:, :, center_h:center_h + s,
                                    center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            cuda_zero = ch.tensor(0).cuda()
            new_deltas = new_deltas / ch.sqrt(ch.sum(new_deltas ** 2, dim=(2, 3), keepdim=True)) * (
                ch.maximum(eps ** 2 - curr_norms_image ** 2, cuda_zero) / c + norms_windows ** 2) ** 0.5
            delta_curr[:, :, center_h_2:center_h_2 + s2,
                       center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            delta_curr[:, :, center_h:center_h + s,
                       center_w:center_w + s] = new_deltas + 0  # update window_1

            hps_str = 's={}->{}'.format(s_init, s)

            # further modified to work with x_adv
            # x_new = x_curr + delta_curr / \
            #     ch.sqrt(ch.sum(delta_curr ** 2,
            #             dim=(1, 2, 3), keepdim=True)) * eps
            x_new = x_adv_loc_curr + delta_curr / \
                ch.sqrt(ch.sum(delta_curr ** 2,
                        dim=(1, 2, 3), keepdim=True)) * eps
            diff_to_x = x_new - x_curr
            x_new = x_curr + diff_to_x / \
                ch.sqrt(ch.sum(diff_to_x ** 2,
                        dim=(1, 2, 3), keepdim=True)) * eps
            x_new = ch.clip(x_new, x_min, x_max)

            # further modified to work with x_adv
            # curr_norms_image = ch.sqrt(
            #     ch.sum((x_new - x_curr) ** 2, dim=(1, 2, 3), keepdim=True))
            curr_norms_image = ch.sqrt(
                ch.sum((x_new - x_adv_loc_curr) ** 2, dim=(1, 2, 3), keepdim=True))

            logits = self.model.forward(x_new, detach=True)
            # probs = ch.softmax(logits, 1)
            loss = get_loss_fn(self.loss_type, 'none')(logits, y_curr)
            margin = get_loss_fn('margin', 'none')(logits, y_curr)

            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * \
                loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * \
                margin + ~idx_improved * margin_min_curr

            idx_improved = ch.reshape(
                idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * \
                x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            # acc = (margin_min > 0.0).sum() / n_ex_total
            # acc_corr = (1. * (margin_min > 0.0)).mean()
            asr = (margin_min > 0.0).sum() / n_ex_total
            asr_corr = (1. * (margin_min > 0.0)).mean()
            mean_nq, mean_nq_ae, median_nq, median_nq_ae = ch.mean(n_queries), ch.mean(
                n_queries[margin_min <= 0]), ch.median(n_queries), ch.median(n_queries[margin_min <= 0])

            time_total = time.time() - time_start
            self.logger.log(
                '{}: asr={:.2%} asr_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
                format(i_iter + 1, asr, asr_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                       ch.mean(margin_min), ch.amax(curr_norms_image), ch.sum(idx_improved)))
            
            '''
            print('{}: asr={:.2%} asr_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} {}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
                format(i_iter + 1, asr, asr_corr, mean_nq_ae, median_nq_ae, hps_str, x.shape[0], time_total,
                       ch.mean(margin_min), ch.amax(curr_norms_image), ch.sum(idx_improved)))

            '''
            if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == n_iters or asr == 1:
                # TODO: Make sure right things are being logged
                self.logger.add_result(i_iter + 1, {
                    "asr": asr.item(),
                    "asr_corr": asr_corr.item(),
                    "mean_nq": mean_nq.item(),
                    "mean_nq_ae": mean_nq_ae.item(),
                    "median_nq": median_nq.item(),
                    "mean_margin_min": margin_min.mean().item(),
                    "time_total": time_total,
                })
            if asr == 1:
                curr_norms_image = ch.sqrt(
                    ch.sum((x_best - x) ** 2, dim=(1, 2, 3), keepdim=True))
                self.logger.log('Maximal norm of the perturbations: {:.5f}'.format(
                    ch.amax(curr_norms_image)))
                break

        curr_norms_image = ch.sqrt(
            ch.sum((x_best - x) ** 2, dim=(1, 2, 3), keepdim=True))
        self.logger.log('Maximal norm of the perturbations: {:.5f}'.format(
            ch.amax(curr_norms_image)))
        '''
        print('Maximal norm of the perturbations: {:.5f}'.format(
            ch.amax(curr_norms_image)))
        '''
        return n_queries, x_best

    def square_attack_linf(self, x, x_adv_loc, y, corr_classified, n_iters, p_init, rand_start=True):
        """ The Linf square attack """
        eps = self.eps/255
        
        #print(eps)
        if self.seed is not None:
            ch.random.seed(self.seed)
            # ch.random.seed(0)  # important to leave it here as well
        x_min, x_max = 0, 1 if x.max() <= 1 else 255
        c, h, w = x.shape[1:]
        #print(c,h,w)
        n_features = c*h*w
        n_ex_total = x.shape[0]
        #print(y)
        x, x_adv_loc, y = x[corr_classified], x_adv_loc[corr_classified], y[corr_classified]

        # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
        if rand_start:
            init_delta = self._workaround_choice([x.shape[0], c, 1, w], eps)
            #print(init_delta,init_delta.shape)
        else:
            init_delta = ch.zeros([x.shape[0], c, 1, w])
            
        # x_best = ch.clip(x + init_delta, x_min, x_max)
        x_best = ch.clip(ch.clip(x_adv_loc + init_delta, x - eps, x + eps), x_min, x_max)
        #print(x_best.shape)
        logits = self.model.forward(x_best, detach=True)
        #print(logits.shape,y.shape)
        # probs = ch.softmax(logits, 1)
        loss_min = get_loss_fn(self.loss_type, 'none')(logits, y)
        if not self.targeted:
            loss_min =loss_min * -1
        #print(loss_min)
        margin_min = get_loss_fn('margin', 'none')(logits, y, self.targeted)
        # ones because we have already used 1 query
        n_queries = ch.ones(x.shape[0]).cuda()
        loss_=loss_min
        time_start = time.time()
        for i_iter in tqdm(range(n_iters - 1)):
            idx_to_fool = margin_min >0
            

            #print(idx_to_fool)
            #print(margin_min,idx_to_fool)
            x_curr, x_adv_loc_curr, x_best_curr, y_curr = x[idx_to_fool], x_adv_loc[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
            #print("x_best_occ",x_best_curr)
            loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
            # deltas = x_best_curr - x_curr
            deltas = x_best_curr - x_adv_loc_curr

            p = self.p_selection(p_init, i_iter, n_iters)
            #print("A",x_best_curr)
            for i_img in range(x_best_curr.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                s = min(max(s, 1), h-1)
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_curr_window = x_curr[i_img, :,center_h:center_h+s, center_w:center_w+s]
                #print(x_curr_window,x_curr_window.shape)
                x_adv_loc_curr_window = x_adv_loc_curr[i_img, :,center_h:center_h+s, center_w:center_w+s] 
                x_best_curr_window = x_best_curr[i_img, :,center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                """
                while ch.sum(ch.abs(ch.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], x_min, x_max) - x_best_curr_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w +
                           s] = self._workaround_choice([c, 1, 1], eps)
                """
                #print((x_adv_loc_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s]).shape)

                while ch.sum(ch.abs(ch.clip(ch.clip(x_adv_loc_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s],x_curr_window-eps,x_curr_window+eps), x_min, x_max) - x_best_curr_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w +
                           s] = self._workaround_choice([c, 1, 1], eps)
                    #print(deltas,deltas.shape)
                for id in range(10):
                    if not idx_to_fool[id]:
                        loss_[id]=loss_min[id]
            # modified to work with x_adv_loc
            # x_new = ch.clip(x_curr + deltas, x_min, x_max)
            #print(deltas[0].sum())
            x_new = ch.clip(ch.clip(x_adv_loc_curr+deltas,x_curr-eps,x_curr+eps), x_min, x_max)

            logits = self.model.forward(x_new, detach=True)
            # probs = ch.softmax(logits, 1)
            loss = get_loss_fn(self.loss_type, 'none')(logits, y_curr)
            #print(y_curr)
            if not self.targeted:
                loss =loss * -1
            #print(loss)
            #print(loss)
            margin = get_loss_fn('margin', 'none')(logits, y_curr, self.targeted)
            #print(margin)

            idx_improved = loss < loss_min_curr

            loss_min[idx_to_fool] = idx_improved * \
                loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = idx_improved * \
                margin + ~idx_improved * margin_min_curr
            idx_improved = ch.reshape(
                idx_improved, [-1, *[1]*len(x.shape[:-1])])
            #print(idx_improved)
            x_best[idx_to_fool] = idx_improved * \
                x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

            # acc = ((margin_min > 0.0).sum()).item() / n_ex_total
            # acc_corr = (1.*(margin_min > 0.0)).mean().item()
            asr = 1.*((margin_min <= 0.0).sum()).item() / n_ex_total
            asr_corr = (1.*(margin_min <= 0.0)).mean().item()

            mean_nq, mean_nq_ae, median_nq_ae = ch.mean(n_queries).item(), ch.mean(
                n_queries[margin_min <= 0]).item(), ch.median(n_queries[margin_min <= 0]).item()
            avg_margin_min = ch.mean(margin_min).item()
            time_total = time.time() - time_start

            self.logger.log('{}: asr={:.2%} asr_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
                  format(i_iter+1, asr, asr_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

            '''
            print('{}: asr={:.2%} asr_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
                  format(i_iter+1, asr, asr_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))
               
            if (i_iter <= 500 and i_iter % 500) or (i_iter > 100 and i_iter % 500) or i_iter + 1 == n_iters or asr == 1:
                self.logger.add_result(i_iter + 1, {
                    "asr": asr,
                    "asr_corr": asr_corr,
                    "mean_nq": mean_nq,
                    "mean_nq_ae": mean_nq_ae,
                    "median_nq_ae": median_nq_ae,
                    "mean_margin_min": margin_min.mean().item(),
                    "time_total": time_total
                })
            '''
            for id in range(10):
                if not idx_to_fool[id]:
                    loss_[id]=loss_min[id]

            if asr == 1:
                break
        #print(x_best)
        #print(n_queries)
        return n_queries, x_best,idx_to_fool,loss_


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Define hyperparameters.')
#     parser.add_argument('--model', type=str, default='pt_resnet', choices=models.all_model_names, help='Model name.')
#     parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2'], help='Attack.')
#     parser.add_argument('--exp_folder', type=str, default='exps', help='Experiment folder to store all output.')
#     parser.add_argument('--gpu', type=str, default='7', help='GPU number. Multiple GPUs are possible for PT models.')
#     parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
#     parser.add_argument('--p', type=float, default=0.05,
#                         help='Probability of changing a coordinate. Note: check the paper for the best values. '
#                              'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
#     parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
#     parser.add_argument('--n_iter', type=int, default=10000)
#     parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
#     args = parser.parse_args()
#     args.loss = 'margin' if not args.targeted else 'cross_entropy'

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     dataset = 'mnist' if 'mnist' in args.model else 'cifar10' if 'cifar10' in args.model else 'imagenet'
#     timestamp = str(datetime.now())[:-7]
#     hps_str = '{} model={} dataset={} attack={} n_ex={} eps={} p={} n_iter={}'.format(
#         timestamp, args.model, dataset, args.attack, args.n_ex, args.eps, args.p, args.n_iter)
#     args.eps = args.eps / 255.0 if dataset == 'imagenet' else args.eps  # for mnist and cifar10 we leave as it is
#     batch_size = data.bs_dict[dataset]
#     model_type = 'pt' if 'pt_' in args.model else 'tf'
#     n_cls = 1000 if dataset == 'imagenet' else 10
#     gpu_memory = 0.5 if dataset == 'mnist' and args.n_ex > 1000 else 0.15 if dataset == 'mnist' else 0.99

#     log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
#     metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)

#     log = utils.Logger(log_path)
#     log.print('All hps: {}'.format(hps_str))

#     if args.model != 'pt_inception':
#         x_test, y_test = data.datasets_dict[dataset](args.n_ex)
#     else:  # exception for inception net on imagenet -- 299x299 images instead of 224x224
#         x_test, y_test = data.datasets_dict[dataset](args.n_ex, size=299)
#     x_test, y_test = x_test[:args.n_ex], y_test[:args.n_ex]

#     if args.model == 'pt_post_avg_cifar10':
#         x_test /= 255.0
#         args.eps = args.eps / 255.0

#     models_class_dict = {'tf': models.ModelTF, 'pt': models.ModelPT}
#     model = models_class_dict[model_type](args.model, batch_size, gpu_memory)

#     logits_clean = model.forward(x_test)
#     corr_classified = logits_clean.argmax(1) == y_test
#     # important to check that the model was restored correctly and the clean accuracy is high
#     log.print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)))

#     square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
#     y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
#     y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
#     # Note: we count the queries only across correctly classified images
#     n_queries, x_adv = square_attack(model, x_test, y_target_onehot, corr_classified, args.eps, args.n_iter,
#                                      args.p, metrics_path, args.targeted, args.loss)

