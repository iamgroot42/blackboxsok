from simple_parsing import ArgumentParser
import os
from bbeval.virustotal import VisusTotal


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--name", help="Name of experiment", type=str)
    args = parser.parse_args()

    loaddir = os.path.join("/p/blackboxsok/malware_samples/", args.name)
    if not os.path.exists(loaddir):
        raise ValueError(f"Path {loaddir} not found")

    # Get paths
    paths = os.listdir(loaddir)
    paths = [os.path.join(loaddir, path) for path in paths]

    # Get preds from VirusTotal
    api = VisusTotal()
    preds = api.get_preds(paths)
    actual_preds = preds[:, 0] / (preds[:, 0] + preds[:, 1])
    print(f"Average evasion rate: {actual_preds.mean()}")
