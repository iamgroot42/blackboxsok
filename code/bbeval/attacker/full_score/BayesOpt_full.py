import numpy as np
import torch
import GPy
from tqdm import tqdm
from bbeval.models.core import GenericModelWrapper
from bbeval.attacker.core import Attacker
from bbeval.config import TransferredAttackConfig, AttackerConfig, ExperimentConfig
from bbeval.loss import get_loss_fn
import time
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import  gen_batch_initial_conditions
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from bbeval.attacker.full_score.BayesOpt_full_util import  proj,latent_proj,fft_transform,fft_transform_mc,transform

class BayesOpt_full(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.params = TransferredAttackConfig(**self.params)
        self.device ="cuda"
        self.eps=self.eps/255
        self.arch = "inception_v3"
        self.inf_norm =True
        self.discrete=True
        self.hard_label=False
        self.dim =12
        self.standardize=True
        self.bounds=0
        self.acqf="EI"
        self.optimize_acq = 'scipy'
        self.initial_samples =5
        self.sin =True
        self.cos = True
        self.beta=1
        self.itr =99

    def obj_func(self, x, x0, y0,target_label=None):
        # evaluate objective function
        # if hard label: -1 if image is correctly classified, 0 otherwise
        # (done this way because BayesOpt assumes we want to maximize)
        # if soft label, correct logit - highest logit other than correct logit
        # in both cases, successful adversarial perturbation iff objective function >= 0
        x = transform(x, self.arch, self.cos, self.sin).to(self.device)
        x = proj(x, self.eps, self.inf_norm, self.discrete)
        x_a=torch.clip(torch.clip(x+x0 , x0 - self.eps, x0 + self.eps), 0, 1)
        with torch.no_grad():
            #self.model.set_eval()
            #self.model.zero_grad()
            y = self.model.forward(x_a)
        #hard-labelblack-box attacks
        #for small query budgets and report
        #success rates and average queries.
        if self.targeted:

            y = torch.log_softmax(y, dim=1)
            #print(y)
            #max_score = y[:, 0]
           

            #print(max_score)
            y, index = torch.sort(y, dim=1, descending=True)
            max_score = y[:, 0]
            #print(y)
            #print(index)
            #elect_index = (index[:, 0] == target_label).long()
            
            idx = (index == target_label).nonzero().flatten()
            #print ("index of target",idx.tolist()[-1]) # [2]


            #next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
            next_max =y[0][idx.tolist()[-1]]
            #print("max score",max_score)
            #print("max index",index[:, 0])
            #print("target score",next_max)
            #print(next_max)
            f = torch.max(max_score-next_max,torch.zeros_like(next_max))
            '''
            train_obj1=[]
            train_obj1.append(f)
            f=torch.tensor(train_obj1)
            f.cuda()
            '''
            print("f",-f)
            # inverse to maxize the negative value
            #print(f.device())
            return -f
        else:

            y = torch.log_softmax(y, dim=1)
            
            max_score = y[:, y0]
            print(max_score)
            y, index = torch.sort(y, dim=1, descending=True)
            select_index = (index[:, 0] == y0).long()
            print(y)
            print(index[:, y0])
            next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
            print(next_max)
            #print(next_max)
            f = torch.max(max_score - next_max, torch.zeros_like(max_score))
            print(-f)
            return -f

    def initialize_model(self,x0, y0, target_label=None,n=1):
        # initialize botorch GP model
        # generate prior xs and ys for GP
        train_x = 2 * torch.rand(n, self.latent_dim, device=self.device).float() - 1
        if not self.inf_norm:
            train_x = latent_proj(train_x, self.eps)
        train_obj = self.obj_func(train_x, x0, y0,target_label)
        '''
        train_obj1=[]
        train_obj1.append(train_obj)
        train_obj=torch.tensor(train_obj1)
        '''
        #print(train_obj,train_obj.shape)
        mean= train_obj.mean()
        std =torch.std(train_obj, unbiased=False)
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

    def optimize_acqf_and_get_observation(self,acq_func, x0, y0,target_label=None):
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
        new_obj = self.obj_func(new_x, x0, y0,target_label)
        return new_x, new_obj

    def bayes_opt(self,x0, y0,target_label = None):
        """
        Main function for Bayesian Optimiazation.
        After initialization of model, fit GP to the data for each iteration
        and get a new point by the acquisition function. Then adds it to data
        to check the success of the attack.
        """
        best_observed = []
        query_count, success = 0, 0
        best_candidate,best_adv_added =[],[]
        #initialization of the GP model
        train_x, train_obj, mll, model, best_value, mean, std = self.initialize_model(
            x0, y0, target_label,n=1)
        best_observed.append(best_value)
        query_count += 1

        #run self.itr rounds for simplicity
        for i in range(self.itr):
            #mll.cuda()
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
            new_x, new_obj = self.optimize_acqf_and_get_observation(qEI, x0, y0,target_label)
            '''
            if args.standardize:
                new_obj = (new_obj - mean) / std
            '''
            # updates data points
            train_x = torch.cat((train_x, new_x))
            #print(train_x.shape,new_obj.shape)
            train_obj = torch.cat((train_obj, new_obj))

            best_value, best_index = train_obj.max(0)
            best_observed.append(best_value.item())
            best_candidate = train_x[best_index]
            torch.cuda.empty_cache()
            model.set_train_data(train_x, train_obj, strict=False)

            # get objective value of best candidate; if we found an adversary, exit
            best_candidate = best_candidate.view(1, -1)
            best_candidate = transform(
                best_candidate,  self.arch, self.cos, self.sin).to(self.device)
            best_candidate = proj(best_candidate, self.eps,
                                  self.inf_norm, self.discrete)
            x_=torch.clip(torch.clip(x0+best_candidate , x0 - self.eps, x0 + self.eps), 0, 1)
            with torch.no_grad():
                #self.model.set_eval()
                #self.model.zero_grad()
                adv_label = torch.argmax(
                    self.model.forward(x_))
            if self.targeted:
                if adv_label == target_label:
                    success = 1
                    '''
                    if self.inf_norm:
                        print('Adversarial Label', adv_label.item(),'Norm:', best_candidate.abs().max().item())
                    else:
                        print('Adversarial Label', adv_label.item(),'Norm:', best_candidate.norm().item())
                    '''
                    return query_count, success,best_candidate,best_candidate+x0
                query_count += 1
            else:
            # to check success
                if adv_label != y0:
                    success = 1
                    '''
                    if self.inf_norm:
                        print('Adversarial Label', adv_label.item(),'Norm:', best_candidate.abs().max().item())
                    else:
                        print('Adversarial Label', adv_label.item(),'Norm:', best_candidate.norm().item())
                    '''
                    return query_count, success,best_candidate,best_candidate+x0
                query_count += 1

        return query_count, success,best_candidate,best_adv_added


    def attack(self, x_orig, x_adv,  y_label,  y_target):
        print(y_target)
        suc_num = 0
        time_start = time.time()
        if self.sin and self.cos:
            self.latent_dim = self.dim * self.dim * 3 * 2
        else:
            self.latent_dim = self.dim * self.dim * 3

        self.bounds = torch.tensor([[-2.0] * self.latent_dim, [2.0] * self.latent_dim],
                                   device=self.device).float()

        print("Length of sample_set: ", x_orig.size())
        print("Length of y_label: ", y_label.size())
        results_dict = {}

        adv_dic = {}
        x = 0
        x_sample_adv=[]
        for idx in range(len(x_orig)):
            print("###################===================####################")
            image, label = x_orig[idx], y_label[idx]
            print(label)
            if self.targeted:
                target_label = y_target[idx].int()
                print(target_label)
            #print(image,label)
            image = image.unsqueeze(0).to(self.device)
            #print(image, label)
            self.model.set_eval()
            self.model.zero_grad()
            print(f"Image {idx:d}   Original label: {label:d}")
            predicted_label = torch.argmax(self.model.forward(image))
            print("Predicted label: ", predicted_label.item())
            transfer_flag = False
            success=0
            if predicted_label == label:
                x += 1
            # ignore incorrectly classified images
                # itr, success = bayes_opt(image, label)
                if self.targeted:
                    print(image.shape)
                    itr, success, adv, adv_added_image = self.bayes_opt(
                        image, label,target_label)

                    print(itr, success)
                    if success:
                        results_dict[int((y_label[idx]).detach())] = itr+1
                        adv_dic[idx] = adv, adv_added_image
                        suc_num+=1
                    else:
                        results_dict[int((y_label[idx]).detach())] = self.itr+1
                    x_sample_adv.append(adv_added_image)
                    self.logger.add_result(int(label.detach()), {
                            "query": int(itr),
                            "transfer_flag": int(transfer_flag),
                            "attack_flag": int(success)
                        })
                else:
                        
                    itr, success, adv, adv_added_image = self.bayes_opt(
                        image, label)

                    print(itr, success)
                    if success:
                        results_dict[int((y_label[idx]).detach())] = itr+1
                        adv_dic[idx] = adv, adv_added_image
                        suc_num+=1
                    else:
                        results_dict[int((y_label[idx]).detach())] = self.itr+1
                    x_sample_adv.append(adv_added_image)
                    self.logger.add_result(int(label.detach()), {
                            "query": int(itr),
                            "transfer_flag": int(transfer_flag),
                            "attack_flag": int(success)
                        })
                
            else:
                x_sample_adv.append(x_orig[idx])
                results_dict[int((y_label[idx]).detach())] = 1
                transfer_flag=True
                self.logger.add_result(int(label.detach()), {
                        "query": int(self.itr),
                        "transfer_flag": int(transfer_flag),
                        "attack_flag": int(success)
                    })
            
        #print(x,"images haven been successfully attacked")
        #print('RESULTS', results_dict)

        
        
        query_count = 0
        for idx in results_dict.keys():
            query_count+=results_dict[idx]
                

        time_end = time.time()
        print("\n\nTotal running time: %.4f seconds\n" %
              (time_end - time_start))
        ave_query = 0
        if suc_num != 0:
            ave_query = query_count/len(x_orig)
        x_sample_adv.append(suc_num)
        if x!=0:
            print(x,"image untransfered,",suc_num," success under BayesOpt Attack, net average query of the entire attack is", query_count/self.itr," and average non-transfered image query is", (query_count-len(x_orig))/x)
        else: 
            print("all transfered")
        self.logger.add_result("Final Result", {
                        "success": int(suc_num),
                        "image_avai": int(x),
                        "average query": int((query_count-len(x_orig))/x)
                    })       
        
        return x_sample_adv, query_count+len(x_orig)