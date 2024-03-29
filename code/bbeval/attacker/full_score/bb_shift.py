from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from secml_malware.attack.blackbox.c_black_box_format_exploit_evasion import CBlackBoxContentShiftingEvasionProblem
from secml.array import CArray
import numpy as np
from typing import List
import copy

from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from bbeval.models.pytorch.malware import SecmlEnsemblPhi


class BlackboxSectionInjection(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        wrapper = self.model.get_phi_wrapper_class()
        self.phi_net = wrapper(self.model.model)
        self.local_phi_net = self.phi_net
        if aux_models is not None and len(aux_models) > 0:
            self.local_phi_net = SecmlEnsemblPhi(list(aux_models.values()))

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):
        """
            If aux models present, use them for crafting examples.
            Else, use the 'target' model.
        """
        population_size = self.params['population_size']  # 10
        penalty_regularizer = self.params['penalty_regularizer']  # 0
        iterations = self.params['iterations']  # 100
        bytes_to_inject = self.params['bytes_to_inject']  # 512

        attack = CBlackBoxContentShiftingEvasionProblem(
                    self.local_phi_net,
                    population_size=population_size,
                    penalty_regularizer=penalty_regularizer,
                    iterations=iterations,
                    bytes_to_inject=bytes_to_inject)
        engine = CGeneticAlgorithm(attack)

        x_adv_new = []
        results = []
        for i, (x_orig_i, x_adv_i) in enumerate(zip(x_orig, x_adv)):
            x_adv_i_feature = CArray(np.frombuffer(x_adv_i.bytes, dtype=np.uint8)).atleast_2d()
            x_adv_i.feature = x_adv_i_feature
            y_pred, adv_score, adv_ds, f_obj = engine.run(
                x_adv_i.feature, CArray(y_label[i][1].cpu()))
            results.append(adv_score.tondarray()[0][1])
            real_adv_x = adv_ds.X[0, :].tolist()[0]
            real_adv_x = b''.join([bytes([i]) for i in real_adv_x])
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