from simple_parsing import ArgumentParser
import os
from bbeval.virustotal import VisusTotal


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--name", help="Name of experiment", type=str)
    args = parser.parse_args()

    loaddir = os.path.join("/p/blackboxsok/malware_samples_all/", args.name)
    if not os.path.exists(loaddir):
        raise ValueError(f"Path {loaddir} not found")

    # Get paths
    paths = os.listdir(loaddir)
    paths = [os.path.join(loaddir, path) for path in paths]

    # Get preds from VirusTotal
    api = VisusTotal()
    preds = api.get_preds(paths)
    actual_preds = preds[:, 0] / (preds[:, 0] + preds[:, 1])
    cutoffs = [1, 5, 10, 15, 20, 25, 30, 35]
    num_evasions = [(preds[:, 0] <= x).mean() for x in cutoffs]
    print(f"Average evasion rate: {actual_preds.mean()}")
    for cutoff, num_evasion in zip(cutoffs, num_evasions):
        print(f"Sample evasion rate for <={cutoff} detections: {num_evasion}")

    # Save predictions as well
    save_file = f"./vt_outputs/{args.name}.txt"
    with open(save_file, "w") as f:
        for i in range(len(preds)):
            f.write(str(preds[i][0]) + "," + str(preds[i][1]) + "\n")
