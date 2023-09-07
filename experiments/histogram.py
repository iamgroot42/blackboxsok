import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def get_evasions_falues(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            vals.append(float(line.strip().split(",")[0]))
    return np.array(vals)


if __name__ == "__main__":
    # names = ["okreuk_sorel_malconv", "okreukvmi_sorel_malconv", "kreukmi_sorel_malconv", "nkreuk2_sorel_malconv", "kreuk_sorel_malconv"]
    # labels =["FGSM", "FGSM-VMI", "FGSM-MI", "FGSM(2)", "FGSM (og-t)"]
    # names = ["kreuk_sorel_malconv", "kreukmomentum_sorel_malconv", "kreukvmi_sorel_malconv"]
    # names = ["okreuk_sorel_malconv", "okreukvmi_sorel_malconv", "kreukmi_sorel_malconv"]
    # labels =["FGSM", "FGSM-MI", "FGSM-VMI"]
    #names = ["dosextend_malconv", "dosextend_sorel", "dosextend_gbt", "dosextend_malconv_gbtall_sorel_all", "dosextend_malconv_gbt", "dosextend_malconv_sorel", "dosextend_sorelall", "dosextend_gbt_all", "dosextend_malconv_gbtall_sorel", "dosextend_malconv_gbt_sorel_all"]
    #labels = ["Malconv", "SOREL", "GBT", "MalConv_GBTALL_SORELALL", "Malconv_GBT", "Malconv_SOREL", "SORELALL", "GBTALL", "Malconv_GBTALL_SOREL", "Mallconv_GBT_SORELALL"]
    names = ["dosextend_malconv_gbtall_sorel_all"]
    labels = ["i-Ensemble"]
    vals = [get_evasions_falues(f"./vt_outputs/{name}.txt") for name in names]
    # Plot histogram
    for i, val in enumerate(vals):
        plt.hist(val, bins=[1, 5, 10, 15, 20, 25, 30, 35], label=labels[i], alpha=0.5)
        # print("Average # of evasions for {}: {}".format(labels[i], val.mean()))
        print("Average % of evasions for {} (>= 15 evasions): {}".format(labels[i], 100 * (val >= 15).mean()))
        print("Average % of evasions for {} (>= 30 evasions): {}".format(labels[i], 100 * (val >= 30).mean()))
    # Save plot
    plt.xlabel("Number of classifiers evaded (out of 73)")
    plt.savefig(f"histogram.png")
