import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.autograd import Variable as V

from bbeval.attacker.core import Attacker
from bbeval.config import NESConfig, AttackerConfig, ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.loss import get_loss_fn
from bbeval.attacker.transfer_methods._manipulate_input import clip_by_tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import  gen_batch_initial_conditions
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from bbeval.attacker.full_score.BayesOpt_full_util import  proj,latent_proj,fft_transform,fft_transform_mc,transform
np.set_printoptions(precision=5, suppress=True)



class NES_bayes(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        # Parse params dict into SquareAttackConfig
        self.params = NESConfig(**self.params)
        self.x_final = None
        self.queries = 1
        self.arch = "inception_v3"
        self.discrete=True
        self.hard_label=False
        self.dim =12
        self.inf_norm =True
        self.standardize=True
        self.bounds=0
        self.acqf="EI"
        self.optimize_acq = 'scipy'
        self.initial_samples =5
        self.sin =True
        self.cos = True
        self.beta=1
        self.device ="cuda"
####    TO DO: change the loss function
        self.criterion = get_loss_fn("scel")
        self.norm = None
        self.k = 1
    def obj_func(self, x,x0,target_label=None):
        # evaluate objective function
        # if hard label: -1 if image is correctly classified, 0 otherwise
        # (done this way because BayesOpt assumes we want to maximize)
        # if soft label, correct logit - highest logit other than correct logit
        # in both cases, successful adversarial perturbation iff objective function >= 0
        x = transform(x, self.arch, self.cos, self.sin).to(self.device)
        x = proj(x, self.eps, self.inf_norm, self.discrete)
        with ch.no_grad():
            #self.model.set_eval()
            #self.model.zero_grad()
            y = self.model.forward(x +x0)
        #hard-labelblack-box attacks
        #for small query budgets and report
        #success rates and average queries.

        y = ch.log_softmax(y, dim=1)
        #print(y)
        #max_score = y[:, 0]
        

        #print(max_score)
        y, index = ch.sort(y, dim=1, descending=True)
        max_score = y[:, 0]
        #print(y)
        #rint(index)
        #elect_index = (index[:, 0] == target_label).long()
        
        idx = (index == target_label).nonzero().flatten()
        #print ("index of target",idx.tolist()[-1]) # [2]


        #next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
        next_max =y[0][idx.tolist()[-1]]
        #print("max score",max_score)
        #print("max index",index[:, 0])
        #print("target score",next_max)
        #print(next_max)
        f = ch.max(max_score-next_max,ch.zeros_like(next_max))
        
        #print("f",-f)

        return -f

    def initialize_model(self,x0,target_label=None,n=1):
        # initialize botorch GP model
        # generate prior xs and ys for GP
        
        train_x = 2 * ch.rand(n, self.latent_dim, device=self.device).float() - 1
        if not self.inf_norm:
            train_x = latent_proj(train_x, self.eps)
        

        train_obj = self.obj_func(train_x,x0,target_label)
        mean= train_obj.mean()
        std =ch.std(train_obj, unbiased=False)
        '''
        if self.standardize:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()
        '''
        best_observed_value = train_obj.max().item()
        # define models for objective and constraint
        model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
        model = model.to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = mll.to(train_x)
        return train_x, train_obj, mll, model, best_observed_value, mean, std

    def optimize_acqf_and_get_observation(self,x0,acq_func,target_label=None):
        # Optimizes the acquisition function, returns new candidate new_x
        # and its objective function value new_obj

        # optimize( old botorch doesnt support optimize_acqf anymore
        '''
        if args.optimize_acq == 'scipy':
            candidates = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=args.q,
                num_restarts=args.num_restarts,
                raw_samples=200,
                sequential =False
            )
        else:
        '''
        Xinit = gen_batch_initial_conditions(
            acq_func,
            self.bounds,
            q=1,
            num_restarts=1,
            raw_samples=500
        )
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=Xinit,
            acquisition_function=acq_func,
            lower_bounds=self.bounds[0],
            upper_bounds=self.bounds[1],
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)

        # observe new values
        new_x = candidates.detach()
        if not self.inf_norm:
            new_x = latent_proj(new_x, self.eps)
        new_obj = self.obj_func(new_x,x0,target_label)
        return new_x, new_obj

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
        self.eps = self.eps / 255.0
        eps=self.eps
        targeted = self.targeted
        x_min_val, x_max_val = 0, 1.0
        momentum = 0.9
        samples_per_draw = 100
        is_preturbed = False
        sigma = 1e-5
        batch_size = 10
        max_queries = 100000
        min_lr = 1e-3
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
        if self.sin and self.cos:
            self.latent_dim = self.dim * self.dim * 3 * 2
        else:
            self.latent_dim = self.dim * self.dim * 3

        self.bounds = ch.tensor([[-2.0] * self.latent_dim, [2.0] * self.latent_dim],
                                   device=self.device).float()

        

        

        for idx in range(len(x_orig)):
            temp_eps = 1

            print("###################===================####################")
            print(idx)

            stop_queries = 0
            x_image, initial_adv, target_label = x_orig[idx].unsqueeze(0),x_target[idx].unsqueeze(0), y_target[idx].int()
            print(x_image.shape)
            lower = clip_by_tensor(x_image - temp_eps, x_min_val, x_max_val)
            upper = clip_by_tensor(x_image + temp_eps, x_min_val, x_max_val)
            adv = clip_by_tensor(initial_adv, lower, upper)
            self.model.set_eval()  # Make sure model is in eval model
            self.model.zero_grad()  # Make sure no leftover gradients
            predicted_label = ch.argmax(self.model.forward(x_image))



            best_observed = []
            query_count, success = 0, 0
            best_candidate,best_adv_added =[],[]
            #initialization of the GP model
            train_x, train_obj, mll, model, best_value, mean, std = self.initialize_model(
                x_image, target_label,n=1)
            best_observed.append(best_value)

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
                    break
                if stop_queries + 1 > max_queries:
                    print("Out of queries!")
                    break
                num_queries+=1
                stop_queries+=1




                # fit the model
                fit_gpytorch_model(mll)
                # define the qNEI acquisition module
                if self.acqf == 'EI':
                    qEI = ExpectedImprovement(model=model, best_f=best_value)
                elif self.acqf == 'PM':
                    qEI = PosteriorMean(model)
                elif self.acqf == 'POI':
                    qEI = ProbabilityOfImprovement(model, best_f=best_value)
                elif self.acqf == 'UCB':
                    qEI = UpperConfidenceBound(model, beta=self.beta)
                # optimize and get new observation
                new_x, new_obj = self.optimize_acqf_and_get_observation(x_image,qEI, target_label)    
                    
                l= new_obj
                print(l)
                print("Current label: " + str(target_model_prediction.item()) + "   loss: " + str(l.item())+"   eps: "+ str(temp_eps))


                train_x = ch.cat((train_x, new_x))
                train_obj = ch.cat((train_obj, new_obj))

                best_value, best_index = train_obj.max(0)
                best_observed.append(best_value.item())
                best_candidate = train_x[best_index]
                ch.cuda.empty_cache()
                model.set_train_data(train_x, train_obj, strict=False)

                # get objective value of best candidate; if we found an adversary, exit
                best_candidate = best_candidate.view(1, -1)
                best_candidate = transform(
                    best_candidate,  self.arch, self.cos, self.sin).to(self.device)
                best_candidate = proj(best_candidate, self.eps,
                                    self.inf_norm, self.discrete)


                perturbation=best_candidate
                #print(perturbation)

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

                adv_thresh=0#need to modify
                if l < adv_thresh and temp_eps > eps:
                    prop_de = delta_epsilon

                while current_lr >= min_lr:
                    # PARTIAL INFORMATION ONLY
                    proposed_epsilon = max(temp_eps - prop_de, eps)
                    lower = clip_by_tensor(x_image - proposed_epsilon, x_min_val, x_max_val)
                    upper = clip_by_tensor(x_image + proposed_epsilon, x_min_val, x_max_val)
                    # GENERAL LINE SEARCH
                    proposed_adv = adv.cpu() - targeted * current_lr * perturbation.cpu()
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
