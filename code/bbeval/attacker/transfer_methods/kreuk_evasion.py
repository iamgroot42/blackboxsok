from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from secml_malware.attack.whitebox import CKreukEvasion
from secml_malware.models.c_classifier_end2end_malware import End2EndModel
from secml.array import CArray
from typing import List
import copy


class Padding(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
    
    def _attack(self, 
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper]):
        padding_bytes = self.params['how_many_padding_bytes'] # 2048
        iterations = self.params['iterations'] # 5
        epsilon = self.params['epsilon'] # 1.0
        fgsm = CKreukEvasion(self.model.model,
                             how_many_padding_bytes=padding_bytes,
                             epsilon=epsilon,
                             iterations=iterations)
        x_adv_new = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):

            x_adv_i_feature = End2EndModel.bytes_to_numpy(
                x_adv_i.bytes, self.model.model.get_input_max_length(), 256, False
            )

            y_pred, adv_score, adv_ds, f_obj = fgsm.run(CArray(x_adv_i_feature), CArray(label[1]))
            real_adv_x = fgsm.create_real_sample_from_adv(x_orig_i.path, adv_ds.X)
            x_adv_i_new: MalwareDatumWrapper  = copy.deepcopy(x_orig)
            x_adv_i_new.bytes = real_adv_x
            x_adv_new.append(x_adv_new)

        stop_queries = 1

        return real_adv_x, stop_queries
