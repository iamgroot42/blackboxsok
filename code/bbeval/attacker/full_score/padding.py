from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper
from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaEvasionProblem
from secml.array import CArray
import numpy as np
from typing import List
import copy

from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from bbeval.models.pytorch.malware import SecmlEnsemblPhi


class GammaPadding(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self.goodware_path = "/p/blackboxsok/datasets/phd-dataset/combined"
        wrapper = self.model.get_phi_wrapper_class()
        self.phi_net = wrapper(self.model.model)
        self.threshold = self.model.threshold
        self.local_phi_net = self.phi_net
        if aux_models is not None and len(aux_models) > 0:
            self.local_phi_net = SecmlEnsemblPhi(list(aux_models.values()))
            self.threshold = self.local_phi_net.threshold

    def _prepare_goodware(self, how_many: int):
        section_population, what_from_who = CGammaEvasionProblem.create_section_population_from_folder(self.goodware_path,
                                                how_many=how_many,
                                                sections_to_extract=['.rdata'])
        return section_population

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
        penalty_regularizer = self.params['penalty_regularizer']  # 1e-6
        iterations = self.params['iterations']  # 10
        how_many = self.params['how_many']  # 100

        section_population = self._prepare_goodware(how_many)

        attack = CGammaEvasionProblem(section_population,
                                      self.local_phi_net,
                                      population_size=population_size,
                                      penalty_regularizer=penalty_regularizer,
                                      iterations=iterations,
                                      threshold=self.threshold)
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