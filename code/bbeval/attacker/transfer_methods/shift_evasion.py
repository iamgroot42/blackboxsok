from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from secml_malware.attack.whitebox.c_shift_evasion import CFormatExploitEvasion
from secml_malware.models.c_classifier_end2end_malware import End2EndModel
from secml.array import CArray
from typing import List
import copy


class Shift(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):
        preferable_extension_amount = self.params['preferable_extension_amount']  # 512
        iterations = self.params['iterations']  # 100
        shift = CFormatExploitEvasion(self.model.model,
                                      preferable_extension_amount=preferable_extension_amount,
                                      iterations=iterations)
        x_adv_new = []
        results = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):
            x_adv_i_feature = End2EndModel.bytes_to_numpy(
                x_adv_i.bytes, self.model.model.get_input_max_length(), 256, False
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