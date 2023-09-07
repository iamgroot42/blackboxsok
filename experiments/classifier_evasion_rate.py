"""
    Test detection rate for specified classifier, using samples
    loaded from specified directory
"""
from simple_parsing import ArgumentParser
from pathlib import Path
import os
from bbeval.config import ModelConfig
from bbeval.models.utils import get_model_wrapper
from bbeval.datasets.malware.base import MalwareDatumWrapper
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--load_exp_name", help="Name of experiment", type=str)
    parser.add_argument(
        "--config", help="Specify config file", type=Path)
    args = parser.parse_args()
    model_config = ModelConfig.load(args.config, drop_extra_fields=False)

    loaddir = os.path.join("/p/blackboxsok/malware_samples_all/", args.load_exp_name)
    if not os.path.exists(loaddir):
        raise ValueError(f"Path {loaddir} not found")

    # Get paths
    paths = os.listdir(loaddir)
    paths = [os.path.join(loaddir, path) for path in paths]

    # Load up model(s)
    target_model = get_model_wrapper(model_config)
    target_model.cuda()
    num_misclassified = 0
    num_so_far = 0
    iterator = tqdm(paths)
    preds = []
    for path in iterator:
        x_sample = MalwareDatumWrapper(path)
        y_label_adv = target_model.predict_proba([x_sample])[0, :]
        prediction = 1 * (y_label_adv[1].item() >= target_model.threshold)
        num_misclassified += (prediction == 0)
        num_so_far += 1
        running_evasion_rate = num_misclassified / num_so_far
        preds.append(prediction)
        iterator.set_description("Evasion rate: %.1f%%" % (100 * running_evasion_rate))

    print(",".join([str(x) for x in preds]))
