from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.attacker.core import Attacker

import magic
from secml.array import CArray

from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion


class HeaderEvasion(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self._init_params()
        self._define_attacker()
    
    def _init_params(self):
        self.iterations = 50
        self.random_init = False
        self.optimize_all_dos = False
        self.threshold = 0.5
    
    def _define_attacker(self):
        self.attack_obj = CHeaderEvasion(
            self.model,
            random_init=self.random_init,
            iterations=self.iterations,
            optimize_all_dos=self.optimize_all_dos,
            threshold=self.threshold)

    def _attack(self, x_orig, x_adv, y_label, y_target=None):
        x_adv_data, x_adv_filenames = x_adv
        adv_samples = []
        for i, (sample, label) in enumerate(zip(x_adv_data, y_label)):
            # Launch attack
            y_pred, adv_score, adv_ds, f_obj = self.attack_obj.run(
                CArray(sample), CArray(label[1]))
            
            # Consruct adversarial example
            adv_x = adv_ds.X[0,:]
            real_adv_x = self.attack_obj.create_real_sample_from_adv(
                x_adv_filenames[i], adv_x)

            adv_samples.append(real_adv_x)

        # No stopping criteria inside attack
        return adv_samples, self.attack_obj.num_queries_used
