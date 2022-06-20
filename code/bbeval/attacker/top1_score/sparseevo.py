from tqdm import tqdm
import logging
import numpy as np
import torch as ch

from bbeval.attacker.core import Attacker
from bbeval.config import AttackerConfig, SparseEvoConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.attacker.top1_score._spaevo_attack import SpaEvoAtt, l0


np.set_printoptions(precision=5, suppress=True)


class SparseEvo(Attacker):
    def __init__(self, model: GenericModelWrapper, aux_models: dict, config: AttackerConfig):
        super().__init__(model, aux_models, config)
        # Parse params dict into SquareAttackConfig
        self.params = SparseEvoConfig(**self.params)
        self.scale = np.array(self.params.scale)
        # Put model in eval model
        self.model.set_eval()
        self.attacker_obj = SpaEvoAtt(model, self.params.n_pix,
                                      pop_size=self.params.pop_size,
                                      cr=self.params.pop_size,
                                      mu=self.params.mu,
                                      seed=self.seed,
                                      flag=self.targeted)

    def _attack(self, x, y):
        if self.targeted:
            # TODO: Handle this ID_set, testset etc.
            # t = ID_set[i,3] #tID, 3 is index acrross dataset - 4 is sample index in a class (not accross dataset)
            # timg, tlabel = testset[t]
            t = None
            timg, tlabel = None, None
            timg = ch.unsqueeze(timg, 0).cuda()
        else:
            init_mode = 'salt_pepper_att' #'gauss_rand' #'salt_pepper'
            timg, nqry,_ = self._gen_starting_point(x, y, init_mode)
            tlabel = None
            nquery += nqry
        x_adv, num_queries, _ = self.attacker_obj.evo_perturb(
            x, timg, y, tlabel, max_query=self.query_budget)
        return x_adv, num_queries
    
    def _check_adv_status(self, img, olabel, tlabel, flag=True):
        is_adv = False
        pred_label = self.model.predict_label(ch.from_numpy(img).cuda())
        if flag == True:
            if pred_label == tlabel:
                is_adv = True
        else:
            if pred_label != olabel:
                is_adv = True
        return is_adv
    
    # TODO: Consider shifting these 'upscale/noise' functions to somewhere generic
    # May be useful for other attacks as well
    def _rand_img_upscale(self, x, rndtype, scale: int = 8): #(normal/uniform distribution)
        # Set seed
        ch.manual_seed(self.seed)

        # x = [n,c,w,h]
        c = x.shape[1]
        wi = x.shape[2]
        he = x.shape[3]
        x_size = []
        x_size.append(x.shape[0])
        x_size.append(c)
        x_size.append(wi//scale)
        x_size.append(he//scale)
    
        if rndtype == 'normal':
            rnd = ch.randn(x_size).type(ch.FloatTensor)
        else:
            rnd = ch.rand(x_size).type(ch.FloatTensor)

        tx = ch.clamp(rnd.cuda(),0,1)
        tx = tx[0].cpu().numpy()
        tx = np.transpose(tx,(1,2,0))
    
        ox = np.zeros((wi,he,c))
    
        for i in range(wi//scale):
            for j in range(he//scale):
                ox[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = tx[i,j,:]
    
        ox = np.transpose(ox,(2,0,1))
        ox = ch.from_numpy(ox).unsqueeze(0).cuda()
        ox = ox.to(ch.float)
        return ox
    
    def _salt_pepper_noise(self, x, rndtype, scale: int = 8): #(normal/uniform distribution)
        ch.manual_seed(self.seed)
    
        # x = [n,c,w,h]
        c = x.shape[1]
        wi = x.shape[2]
        he = x.shape[3]
        x_size = []
        x_size.append(x.shape[0])
        x_size.append(c)
        x_size.append(wi//scale)
        x_size.append(he//scale)
    
        if rndtype == 'normal':
            rnd = ch.randn(wi,he).type(ch.FloatTensor)
            tx = ch.clamp((rnd>0*1).type(ch.FloatTensor).cuda(),0,1)
        else:
            rnd = ch.rand(wi,he).type(ch.FloatTensor)
            tx = ch.clamp((rnd>0.5*1).type(ch.FloatTensor).cuda(),0,1)
    
        tx = tx.repeat(c,1,1)
        tx = np.transpose(tx.cpu().numpy(),(1,2,0))
        ox = np.zeros((wi,he,c))
    
        for i in range(wi//scale):
            for j in range(he//scale):
                ox[i*scale:(i+1)*scale,j*scale:(j+1)*scale,:] = tx[i,j,:]
    
        ox = np.transpose(ox,(2,0,1))
        ox = ch.from_numpy(ox).unsqueeze(0).cuda()
        ox = ox.to(ch.float)
        return ox
    
    def _salt_pepper_att(self, oimg, olabel, repetitions: int = 10, eps: float = 0.1):
        min_ = 0
        max_ = 1
        flag = False
        tlabel = None
        # axis = a.channel_axis(batch=False)
        # index of channel & color. Eg: [1,3,32,32] -> channels = 1; [3,32,32] -> channels = 0;
        # number of channel of color = 3 if RGB or 1 if gray scale
        start_qry = 0
        end_qry = 0
        nquery = 0
        #D = np.zeros(5000).astype(int)
        D = ch.zeros(1000, dtype=int).cuda()
        axis = 1
        channels = oimg.shape[axis]

        shape = list(oimg.shape)
        shape[axis] = 1
        r = max_ - min_

        epsilons = 100
        pixels = np.prod(shape)

        epsilons = min(epsilons, pixels)

        max_epsilon = 1
        distance = np.inf
        adv = oimg.copy()
        d = 0

        for i in range(repetitions):
            for epsilon in np.linspace(0, max_epsilon, num=epsilons + 10)[1:]:
                p = epsilon  # probability
                u = ch.rand(size=shape)
                u = ch.repeat_interleave(u, channels, axis=axis)

                salt = (u >= 1 - p / 2).astype(oimg.dtype) * r
                pepper = -(u < p / 2).astype(oimg.dtype) * r
                saltpepper = ch.clip(salt + pepper, -eps, eps)
                perturbed = oimg + saltpepper
                perturbed = ch.clip(perturbed, min_, max_)

                temp_dist = l0(oimg, perturbed)

                if temp_dist >= distance:
                    continue
                nquery += 1
                is_adversarial = self._check_adv_status(
                    perturbed, olabel, tlabel, flag)
                if is_adversarial:
                    # higher epsilon usually means larger perturbation, but
                    # this relationship is not strictly monotonic, so we set
                    # the new limit a bit higher than the best one so far
                    distance = temp_dist
                    adv = perturbed
                    max_epsilon = epsilon * 1.2

                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0(oimg, adv)

                    break
            print('i: %d; nqry: %d, pred_label: %d; temp_dist: %2.3f; L0: %2.3f' % (
                i, nquery, self.model.predict(perturbed.cuda()), temp_dist, distance))
        d = l0(oimg, adv)
        D[end_qry:nquery] = d

        return adv, nquery, D[:nquery]

    def _gen_starting_point(self, x, y,init_mode):
        nqry = 0
        i = 0
        rndtype = 'normal'

        if init_mode == 'salt_pepper_att':
            eps = 1
            repetitions = 2#10
            out, nqry, D = self._salt_pepper_att(x, y, repetitions, eps)
            out = ch.from_numpy(out.reshape(x.shape)).cuda()
        elif init_mode == 'salt_pepper':
            while True:
                out = self._salt_pepper_noise(x, rndtype, self.scale[i])
                label = self.model.predict(out)
                nqry += 1
                if label!= y:
                    #D = np.ones(nqry).astype(int) * l0(x,out)
                    D = ch.ones(nqry,dtype=int).cuda() * l0(x,out)
                    break
                elif i<len(self.scale):
                    i += 1

        elif init_mode == 'gauss_rand':
            while True:
                out = self._rand_img_upscale(x, rndtype, self.scale[i], self.seed)
                label = self.model.predict(out)
                nqry += 1
                if label!= y:
                    #D = np.ones(nqry).astype(int) * l0(x,out)
                    D = ch.ones(nqry,dtype=int).cuda() * l0(x,out)
                    break
                elif i<len(self.scale):
                    i += 1

        return out, nqry, D