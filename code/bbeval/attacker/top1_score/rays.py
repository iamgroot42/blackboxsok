from tqdm import tqdm
import logging
import numpy as np
import torch as ch
from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig
from bbeval.models.core import GenericModelWrapper

np.set_printoptions(precision=5, suppress=True)


class RayS(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig):
        super().__init__(model, aux_models, config)
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
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

    def attack(self, x, x_adv_loc, y, y_target):
        """
            Attack the original image and return adversarial example
            (x, y): original image and label
            x_adv_loc: the starting point for the attack 
            y_target: the target label for the attack, y_target=y for untargeted attacks
        """
        #TODO: implement a version of rays that works for targeted attack, the approach is to 
        # use an image from target class as a guiding direction
        eps = self.eps
        if self.norm_type in [2, np.inf]:
            ord = self.norm_type
        else:
            ord = 'fro'
        
        # Use appropriate labels for the attack
        if self.targeted:
            y_use = y_target
        else:
            y_use = y_label

        shape = list(x_orig.shape)
        dim = np.prod(shape[1:])
        if self.seed is not None:
            np.random.seed(self.seed)

        # init variables
        self.queries = ch.zeros_like(y).cuda()
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
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * \
                block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < self.query_budget)
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            # further modified to work with x_adv_loc
            self.binary_search(x_adv_loc, y_target, attempt, valid_mask)
            # self.binary_search(x, y, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = ch.norm((self.x_final - x_orig).view(shape[0], -1), ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > eps).nonzero().flatten()

            if ch.sum(self.queries >= self.query_budget) == shape[0]:
                self.logger.log('Out of queries', level=logging.WARNING)
                break
            query_string = f"Queries: {ch.min(self.queries.float())}/{self.query_budget}"
            info_string = 'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d' % (ch.mean(
                self.d_t), ch.mean(dist), ch.mean(self.queries.float()), len(working_ind) / len(x_orig), i + 1)
            # Also log all of this information
            # TODO: Make sure right things are being logged
            self.logger.log(query_string + " | " + info_string)
            self.logger.add_result(i + 1, {
                "d_t": ch.mean(self.d_t).item(),
                "adbd": ch.mean(dist).item(),
                "queries": ch.min(self.queries.float()).item(),
                "rob acc": len(working_ind) / len(x_orig),
            })
            iterator.set_description(query_string + " | " + info_string)
            # no need to run till exhaustion for practical bbox attack scenario 
            # so stop early if all AEs are found
            if len(working_ind) / len(x) == 0:
                print("Early stopping due to attack success!")
                break 
        stop_queries = ch.clamp(stop_queries, 0, self.query_budget)
        return self.x_final, stop_queries

    # check whether solution is found
    def search_succ(self, x, y, mask):
        self.queries[mask] += 1
        if self.targeted:
            return self.model.predict(x[mask]) == y[mask]
        else:
            return self.model.predict(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    # further modified to work with x_adv_loc
    def binary_search(self, x_adv_loc, y, sgn, valid_mask, tol=1e-3):
        sgn_norm = ch.norm(sgn.view(len(x_adv_loc), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x_adv_loc), 1, 1, 1)

        d_start = ch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(
            x_adv_loc, sgn_unit, self.d_t), y, valid_mask)
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
