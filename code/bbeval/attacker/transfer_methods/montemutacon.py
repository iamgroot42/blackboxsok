"""
    Based on https://github.com/iamgroot42/montemutacon
"""

from montemutacon_repo.mml.mcts.tree_policy import MctsTreePolicy
from montemutacon_repo.mml.mcts.simulation_policy import MctsSimulationPolicy
from montemutacon_repo.mml.mcts.expansion_policy import MctsExpansionPolicy
from montemutacon_repo.mml.mcts.mcts_mutator import MctsMutator

from montemutacon_repo.tables import mutations_table

from montemutacon_repo.utils.pipeline import Pipeline as CustomPipeline

import dill as pickle
import pandas as pd
import json


pipeline = CustomPipeline(
    "models/surrogate/full_pipeline_surrogate.dat",
    ["models/surrogate/libs_vectorizer_surrogate.dat",
        "models/surrogate/funcs_vectorizer_surrogate.dat"],
)


def classification_function(model, sample) -> int:
    to_convert = sample.copy()
    to_convert["imported_libs"] = [[*to_convert["imported_libs"]]]
    to_convert["imported_funcs"] = [[*to_convert["imported_funcs"]]]
    df = pd.DataFrame.from_dict(to_convert)
    df.drop(columns=["y"], inplace=True)

    # Transform the sample through the pipeline. Depending on your model you might not need this
    transform = pipeline.transform(df, ["imported_libs", "imported_funcs"])
    return model.predict(transform)[0]


if __name__ == '__main__':

    model = pickle.load(open('models/surrogate/trained_tree.dat', 'rb'))
    tree_policy = MctsTreePolicy(2)
    expansion_policy = MctsExpansionPolicy(mutations_table)
    simulation_policy = MctsSimulationPolicy(
        model,
        25,
        expansion_policy,
        mutations_table,
        classification_function,
    )

    mcts_mutator = MctsMutator(
        tree_policy=tree_policy,
        expansion_policy=expansion_policy,
        simulation_policy=simulation_policy,
    )
    samples = json.load(open('samples.json'))
    malware = [sample for sample in samples if sample['y'] == 1]

    results = []
    count = 0

    for sample in malware:
        print(f'Processing sample {count}')
        if classification_function(model, sample) == 0:
            results.append(
                {"skipped": True, "changes": []}
            )

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
        root = mcts_mutator.run(50, sample, starting_state)
        path = mcts_mutator.recover_path(root)

        if path[-1].is_terminal:
            mutations = [node.serialized_option for node in path]
            results.append(
                {"skipped": False, "changes": mutations}
            )
        else:
            results.append(
                {"skipped": False, "changes": []}
            )
        count += 1
