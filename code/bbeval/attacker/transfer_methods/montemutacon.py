# """
#     Based on https://github.com/iamgroot42/montemutacon
# """
#
from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import MalwareAttackerConfig
from bbeval.attacker.core_malware import Attacker
from bbeval.datasets.malware.base import MalwareDatumWrapper, process_with_lief
from bbeval.utils import get_models_save_path

from secml_malware.attack.whitebox import CKreukEvasion
from secml_malware.models.c_classifier_end2end_malware import End2EndModel
from secml.array import CArray
from typing import List

import os
import copy
#
#
# from mml.mcts.tree_policy import MctsTreePolicy
# from mml.mcts.simulation_policy import MctsSimulationPolicy
# from mml.mcts.expansion_policy import MctsExpansionPolicy
# from mml.mcts.mcts_mutator import MctsMutator
#
# from mml.tables import mutations_table
#
# from mml.utils.pipeline import Pipeline as CustomPipeline
#
# import dill as pickle
# import pandas as pd
#
#
class MonteMutacon(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: MalwareAttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)

        base_path = get_models_save_path()
        self.pipeline = CustomPipeline(
            os.path.join(base_path, "montemutacon/surrogate/full_pipeline_surrogate.dat"),
            [
                os.path.join(base_path,"montemutacon/surrogate/libs_vectorizer_surrogate.dat"),
                os.path.join(base_path,"montemutacon/surrogate/funcs_vectorizer_surrogate.dat")]
        )
        self.simulation_depth = 25
        self.exploration_coefficient = 2

        self.model = pickle.load(open(os.path.join(base_path,'montemutacon/surrogate/trained_tree.dat'), 'rb'))
        # print(self.model)
        # exit(0)
        self.tree_policy = MctsTreePolicy(self.exploration_coefficient)
        self.expansion_policy = MctsExpansionPolicy(mutations_table)
        self.simulation_policy = MctsSimulationPolicy(
            self.model,
            self.simulation_depth,
            self.expansion_policy,
            mutations_table,
            self._classification_function,
        )
        self.mcts_mutator = MctsMutator(
            tree_policy=self.tree_policy,
            expansion_policy=self.expansion_policy,
            simulation_policy=self.simulation_policy,
        )
        self.wanted_keys = [
            # "strings_entropy",
            # "num_strings",
            # "file_size",
            # "num_exports",
            # "num_imports",
            # "has_debug",
            # "has_signature",
            # "timestamp",
            # "sizeof_code",
            # "entry",
            # "num_sections",
            "imported_libs",
            "imported_funcs",
        ]
        # Add 'y' as well: 1 for malicious, 0 for benign

    def _classification_function(self, model, sample) -> int:
        to_convert = sample.copy()
        to_convert["imported_libs"] = [[*to_convert["imported_libs"]]]
        to_convert["imported_funcs"] = [[*to_convert["imported_funcs"]]]
        df = pd.DataFrame.from_dict(to_convert)
        df.drop(columns=["y"], inplace=True)

        # Transform the sample through the pipeline. Depending on your model you
        # might not need this
        transform = self.pipeline.transform(df, ["imported_libs", "imported_funcs"])
        return self.model.predict(transform)[0]

    def attack_single(self, sample):
        tried_combinations = {}

        # This is used to keep track of how many times we have performed these
        # changes below. You can add or remove things here to match your setup
        starting_state = {
            "added_strings": 0,
            "removed_strings": 0,
            "added_libs": 0,
            "entropy_changes": 0,
            "tried_combinations": tried_combinations,
        }

        root = self.mcts_mutator.run(50, sample, starting_state)
        path = self.mcts_mutator.recover_path(root)

        if path[-1].is_terminal:
            result = [node.serialized_option for node in path]
        else:
            result = []

        return result

    def _get_relevant_features(self, sample):
        json_features = process_with_lief(sample.bytes, want_json=True)
        json_dict = {
            "strings_entropy": json_features['strings']['entropy'],
            "num_strings": json_features['strings']['numstrings'],
            "file_size": json_features['general']['size'],
            "num_exports": json_features['general']['exports'],
            "num_imports": json_features['general']['imports'],
            "has_debug": json_features['general']['has_debug'],
            "has_signature": json_features['general']['has_signature'],
            "timestamp": json_features['header']['coff']['timestamp'],
            "sizeof_code": json_features['header']['optional']['sizeof_code'],
            "entry": json_features['section']['entry'],
            "num_sections": len(json_features['section']['sections']),
            # "imported_libs": json_features['imports'],
            "imported_libs": [],
            "imported_funcs": [],
            "y": 1
        }
        return json_dict

    def _attack(self,
                x_orig: List[MalwareDatumWrapper],
                x_adv: List[MalwareDatumWrapper],
                y_label=None,
                y_target=None):

        mutations = []
        for sample in x_adv:
            # TODO: Extract "feature" from sample's datum
            sample_json_features = self._get_relevant_features(sample)
            mutation = self.attack_single(sample_json_features)
            mutations.append(mutation)
            exit(0)

        # TODO: Inspect how mutations can be applied to actual file (somewhere in library, perhaps?)

        stop_queries = 1

        self.logger.add_result(
            queries_used=stop_queries,
            result={
                "mutations": mutations
            })

        # TODO- Convert x_adv_new to appropriate batch
        return x_adv_new, stop_queries
