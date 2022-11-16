from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.attacker.core import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from typing import List
from secml_malware.models.c_classifier_end2end_malware import End2EndModel
from secml.array import CArray
import copy

from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion
from bbeval.models.pytorch.malware import SecmlEnsemble


class HeaderEvasion(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.local_model = self.model.model
        if aux_models is not None and len(aux_models) > 0:
            self.local_model = SecmlEnsemble(list(aux_models.values()))

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):
        iterations = self.params['iterations']  # 50
        random_init = self.params['random_init']  # False
        optimize_all_dos = self.params['optimize_all_dos']  # False
        threshold = self.params['threshold']  # 0.5

        shift = CHeaderEvasion(
            self.local_model,
            random_init=random_init,
            iterations=iterations,
            optimize_all_dos=optimize_all_dos,
            threshold=threshold)

        x_adv_new = []
        results = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):
            x_adv_i_feature = End2EndModel.bytes_to_numpy(
                x_adv_i.bytes, self.local_model.get_input_max_length(), 256, False
            )
            x_adv_i.feature = x_adv_i_feature
            y_pred, adv_score, adv_ds, f_obj = shift.run(
                CArray(x_adv_i.feature), CArray(y_label[i][1].cpu()))
            results.append(adv_score.tondarray()[0][1])
            real_adv_x = shift.create_real_sample_from_adv(x_orig_i.bytes, adv_ds.X, input_is_bytes=True)
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
