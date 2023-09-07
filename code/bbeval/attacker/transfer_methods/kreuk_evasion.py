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
import numpy as np

from bbeval.models.pytorch.malware import SecmlEnsemble


class Padding(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.local_model = self.model.model
        self.threshold = self.model.threshold
        if aux_models is not None and len(aux_models) > 0:
            self.local_model = SecmlEnsemble(list(aux_models.values()))
            self.threshold = self.local_model.threshold

        override_threshold = self.params.get('override_threshold', False)
        absolute_nuts = self.params.get('absolute_nuts', False)
        if override_threshold:
            self.threshold = 0.5
        if absolute_nuts:
            self.threshold = 0.0

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):
        padding_bytes = self.params['how_many_padding_bytes']  # 2048
        iterations = self.params['iterations']  # 100
        epsilon = self.params['epsilon']  # 1.0
        momentum = self.params.get('momentum', False)
        variance_tuning = self.params.get('variance_tuning', False)
        p_norm = np.infty if self.params['p_norm'] == 'inf' else 2

        fgsm = CKreukEvasion(self.local_model,
                             how_many_padding_bytes=padding_bytes,
                             epsilon=epsilon,
                             iterations=iterations,
                             momentum_iterative=momentum,
                             variance_tuning=variance_tuning,
                             threshold=self.threshold,
                             p_norm=p_norm)
        x_adv_new = []
        results = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):
            x_adv_i_feature = End2EndModel.bytes_to_numpy(
                x_adv_i.bytes, self.local_model.get_input_max_length(), 256, False
            )
            x_adv_i.feature = x_adv_i_feature
            y_pred, adv_score, adv_ds, f_obj = fgsm.run(
                CArray(x_adv_i.feature), CArray(y_label[i][1].cpu()))
            results.append(adv_score.tondarray()[0][1])
            real_adv_x = fgsm.create_real_sample_from_adv(x_orig_i.bytes, adv_ds.X, input_is_bytes=True)
            x_adv_i_new: MalwareDatumWrapper = copy.deepcopy(x_orig_i)
            x_adv_i_new.bytes = real_adv_x
            x_adv_new.append(x_adv_i_new)

        stop_queries = 1

        self.logger.add_result(
            queries_used=stop_queries,
            result={
                "adv_preds": results
            })

        # TODO- Convert x_adv_new to appropriate batch
        return x_adv_new, stop_queries