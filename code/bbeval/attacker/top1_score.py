import numpy as np
import torch
from core import Attacker

import argparse
import time
# import data
# import models
# import os
# import utils
# from datetime import datetime
np.set_printoptions(precision=5, suppress=True)

class RayS_Attack(Attacker):
    def __init__(self,model,query_budget,norm_type,targeted):
        super().__init__(model,query_budget,norm_type,targeted)

# TODO: decide if we should keep the progress bar as it is.
from pgbar import progress_bar

class RayS(Attacker):
    def __init__(self,model,query_budget,norm_type,targeted):
        super().__init__(model,query_budget,norm_type,targeted)
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.queries = None

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack(self, x, y, eps):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        # TODO: decide if we should have 'str' for order information or number
        if self.norm_type == 'L_2':
            ord = 2
        elif self.norm_type == 'L_Infty':
            ord = np.inf
        else:
            ord = 'fro'

        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if self.seed is not None:
            np.random.seed(self.seed)

        # init variables
        self.queries = torch.zeros_like(y).cuda()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()
        working_ind = (self.d_t > eps).nonzero().flatten()

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
 
        block_level = 0
        block_ind = 0
        for i in range(self.query_budget):
            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < self.query_budget) 
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, self.targeted, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), ord, 1)
            stop_queries[working_ind] = self.queries[working_ind]
            working_ind = (dist > eps).nonzero().flatten()

            if torch.sum(self.queries >= self.query_budget) == shape[0]:
                print('out of queries')
                break
            # TODO: merge this with logging system? 
            progress_bar(torch.min(self.queries.float()), self.query_budget,
                         'd_t: %.4f | adbd: %.4f | queries: %.4f | rob acc: %.4f | iter: %d'
                         % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                            len(working_ind) / len(x), i + 1))
 
        stop_queries = torch.clamp(stop_queries, 0, self.query_budget)
        return self.x_final, stop_queries, dist, (dist <= eps)

    # check whether solution is found
    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if self.targeted:
            return self.model.predict_label(x[mask]) == self.targeted[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    # binary search for decision boundary along sgn direction
    def binary_search(self, x, y, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, self.targeted, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, self.targeted, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]
