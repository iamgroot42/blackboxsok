from tqdm import tqdm
import logging
import numpy as np
import torch as ch
from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper

np.set_printoptions(precision=5, suppress=True)


class RayS(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig, experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.x_final_return = None

        self.queries = None
        # Put model in eval model
        self.model.set_eval()

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = ch.tensor(d).repeat(len(x)).cuda()
        # no modification is needed here because the AE in decision based attacks
        # are initially all require large perturbation
        out = x + d.view(len(x), 1, 1, 1) * v
        out = ch.clamp(out, lb, ub)
        return out

    def attack(self, x_orig, x_adv, y_label,x_target, y_target):
        """
            Attack the original image and return adversarial example
            (x, y): original image and label
            x_adv_loc: the starting point for the attack
            y_target: the target label for the attack, y_target=y for untargeted attacks
        """
        #TODO: implement a version of rays that works for targeted attack, the approach is to
        # use an image from target class as a guiding direction
        self.x_final_return=x_orig.clone()
        stop_queries_return=0
        eps = self.eps/ 255.0
        if self.norm_type in [2, np.inf]:
            ord = self.norm_type
        else:
            ord = 'fro'

        batch_size=50
        batch_num=int(x_orig.shape[0]/batch_size)
        for batch in range(batch_num):
            print("batch: ", batch+1)
            x=x_orig[batch*batch_size: batch*batch_size+batch_size]
            x_adv_loc=x_adv[batch*batch_size: batch*batch_size+batch_size]
            y=y_label[batch*batch_size: batch*batch_size+batch_size]
            x_target_ = x_target[batch*batch_size: batch*batch_size+batch_size]
            y_target_ = y_target[batch*batch_size: batch*batch_size+batch_size]

            shape = list(x.shape)
            dim = np.prod(shape[1:])
            if self.seed is not None:
                np.random.seed(self.seed)

            # init variables
            self.queries = ch.zeros_like(y).cuda()

            if self.targeted:
                # sgn_t is the ray search direction
                self.sgn_t = x_target_ - x_adv_loc
                # d_t is the best decision boundary function given a search direction sgn_t
                self.d_t =  ch.norm(self.sgn_t.view(len(x_adv_loc), -1), 2, 1)
                # print(self.d_t.shape)
                # sys.exit()
            else:
                # sgn_t is the ray search direction
                self.sgn_t = ch.sign(ch.ones(shape)).cuda()
                # d_t is the best decision boundary function given a search direction sgn_t
                self.d_t = ch.ones_like(y).float().fill_(float("Inf")).cuda()

            working_ind = (self.d_t > eps).nonzero().flatten()

            stop_queries = self.queries.clone()
            dist = self.d_t.clone()
            # further modified to work with a_adv_loc
            self.x_final = self.get_xadv(x_adv_loc, self.sgn_t, self.d_t)
            # self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

            block_level = 0
            block_ind = 0
            iterator = tqdm(range(self.query_budget))
            for i in iterator:
                # temp_label1=ch.argmax(self.model.forward(self.x_final[0].unsqueeze(0)))
                # print("result label",temp_label1)

                block_num = 2 ** block_level
                block_size = int(np.ceil(dim / block_num))
                start, end = block_ind * \
                    block_size, min(dim, (block_ind + 1) * block_size)

                valid_mask = (self.queries < self.query_budget)
                attempt = self.sgn_t.clone().view(shape[0], dim)
                attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
                attempt = attempt.view(shape)

                # further modified to work with x_adv_loc
                self.binary_search(x_adv_loc, y_target_, attempt, valid_mask)
                # self.binary_search(x, y, attempt, valid_mask)

                block_ind += 1
                if block_ind == 2 ** block_level or end == dim:
                    block_level += 1
                    block_ind = 0

                dist = ch.norm((self.x_final - x).view(shape[0], -1), ord, 1)
                stop_queries[working_ind] = self.queries[working_ind]
                working_ind = (dist > eps).nonzero().flatten()
                # print(working_ind)

                # raise NotImplementedError(f"The image of {target_modal_name} is not saved yet")

                if ch.sum(self.queries >= self.query_budget) == shape[0]:
                    self.logger.log('Out of queries', level=logging.WARNING)
                    break
                query_string = f"Queries: {ch.min(self.queries.float())}/{self.query_budget}"
                info_string = 'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d' % (ch.mean(
                    self.d_t), ch.mean(dist), ch.mean(self.queries.float()), len(working_ind) / len(x), i + 1)
                iterator.set_description(query_string + " | " + info_string)
                # no need to run till exhaustion for practical bbox attack scenario
                # so stop early if all AEs are found
                if len(working_ind) / len(x) == 0:
                    print("Early stopping due to attack success!")
                    break
            stop_queries = ch.clamp(stop_queries, 0, self.query_budget)
            for idx in range(batch_size):
                self.x_final_return[batch*batch_size +idx]=self.x_final[idx]
                label = y_target_[idx]
                query_c = int(stop_queries[idx])
                if idx not in working_ind:
                    attack_flag = 1
                else:
                    attack_flag = 0

                transfer_flag = 0
                temp_label=ch.argmax(self.model.forward(x_adv_loc[idx].unsqueeze(0)))
                # temp_label1=ch.argmax(self.model.forward(self.x_final[idx].unsqueeze(0)))
                #
                # print("target label",label)
                # print("result label",temp_label1)
                if self.targeted:
                    if temp_label==label:
                        transfer_flag = 1
                        query_c=1
                else:
                    if temp_label!=label:
                        transfer_flag = 1
                        query_c = 1

                self.x_final_return[batch*batch_size +idx]=self.x_final[idx]
                stop_queries_return+=query_c
                self.logger.add_result(int(label.detach()), {
                        "query": int(query_c),
                        "transfer_flag": int(transfer_flag),
                        "attack_flag": int(attack_flag),
                    })
        return self.x_final_return, stop_queries_return

    # check whether solution is found
    def search_succ(self, x, y_target, mask):
        self.queries[mask] += 1
        if self.targeted:
            # print("yes")
            return self.model.predict(x[mask]) == y_target[mask]
        else:
            # print("no")
            return self.model.predict(x[mask]) != y_target[mask]

    # binary search for decision boundary along sgn direction
    # further modified to work with x_adv_loc
    def binary_search(self, x_adv_loc, y, sgn, valid_mask, tol=1e-3):
        sgn_norm = ch.norm(sgn.view(len(x_adv_loc), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x_adv_loc), 1, 1, 1)

        d_start = ch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(
            x_adv_loc, sgn_unit, self.d_t), y, valid_mask)
        # print(initial_succ_mask)
        # print(self.d_t)
        # sys.exit()
        # initial_succ_mask = self.search_succ(self.get_xadv(
        #     x, sgn_unit, self.d_t), y, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = ch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            # further modified to work with x_adv_loc
            search_succ_mask = self.search_succ(self.get_xadv(
                x_adv_loc, sgn_unit, d_mid), y, to_search_ind)
            # search_succ_mask = self.search_succ(self.get_xadv(
            #     x, sgn_unit, d_mid), y, to_search_ind)
            d_end[to_search_ind[search_succ_mask]
                  ] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]
                    ] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[(
                (d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            # further modified to work with x_adv_loc
            self.x_final[to_update_ind] = self.get_xadv(
                x_adv_loc, sgn_unit, d_end)[to_update_ind]
            # self.x_final[to_update_ind] = self.get_xadv(
            #     x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
