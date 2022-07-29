#from tqdm import tqdm
import logging
import numpy as np
import torch as ch
from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig, ExperimentConfig,StairCaseConfig
from bbeval.models.core import GenericModelWrapper


class SimBA(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.params = StairCaseConfig(**self.params)
        self.queries = None
        # Put model in eval model
        self.model.set_eval()

    def normalized_eval(self,x, model):
        if torch.cuda.is_available():
            model.cuda()
            x_copy = x.clone()
            # x_copy = torch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
            #      for i in range(len(x_copy))])
        return model(x_copy)

    def get_probs(self,model, x, y):
        output = self.normalized_eval(x, model).cpu()
        probab = torch.exp(output)
        probab = list(probab.cpu().detach().numpy()[0])
        # print(probab)
        # probab = list(probab.detach().numpy()[0][y])
        return probab[y], probab

    def get_preds(self,model, x):
        output = self.normalized_eval(x, model).cpu()
        probab = torch.exp(output)
        probab = list(probab.cpu().detach().numpy()[0])
        # print(output.data)
        _, preds = output.data.max(1)
        # print(preds)
        return preds
    def _attack(self, x_orig, x, y_origin, y):
        """
        x_orig: original image
        x: attack_seed
        y:target class
        """
        max_queries = 4000
        num_iters = 4000
        log_every = 100
        queries = 0
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        last_prob,log = self.get_probs(self.model, x, y)
        queries += 1
        # print(perm)
        for i in range(num_iters):
            # check if attack succeed
            predicted_y = self.get_preds(self.model, x)
            # print(predicted_y)
            if predicted_y == y: # equal to target class
                print('Iteration %d: queries = %d, prob = %.6f' % (
                    i + 1, queries, last_prob.item()))
                _,log = self.get_probs(self.model, x, y)
                print(log)
                print("Attack succeed")
                sucess_flag = 1
                break

        # if queries >= max_queries or (queries>20000 and last_prob.item()==0):

            if queries >= max_queries:
                print("Attack fails, achieving max queries")
                print(log)
                queries = max_queries
                break

        # craft left example
            x_left = x.clone()
            x_lower = x_orig.view(-1)[perm[i%n_dims]] - epsilon
            x_left = x_left.view(-1)
            x_left[perm[i%n_dims]] = x_lower
            x_left = x_left.view(x.size())

            left_prob,log = self.get_probs(self.model, x_left.clamp(-1, 1), y)
            queries += 1

            if left_prob > last_prob:
                x = x_left.clamp(-1, 1)
                last_prob = left_prob
            else:
                # craft right example
                x_right = x.clone()
                x_upper = x_orig.view(-1)[perm[i%n_dims]] + epsilon
                x_right = x_right.view(-1)
                x_right[perm[i%n_dims]] = x_upper
                x_right = x_right.view(x.size())

                right_prob,log = self.get_probs(self.model, x_right.clamp(-1, 1), y)
                queries += 1
                if right_prob > last_prob:
                    x = x_right.clamp(-1, 1)
                    last_prob = right_prob

            if (i + 1) % log_every == 0 or i == num_iters - 1:
                print('Iteration %d: queries = %d, prob = %.6f' % (
                    i + 1, queries, last_prob.item()))

        return x, queries