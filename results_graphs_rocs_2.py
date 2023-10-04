import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from results_graphs_rocs_1 import measure_name_map, hash_name_map

sets = {}
files = glob.glob("csv_accuracy_results/*_[!_]*.json")
for file in files:
    sets[file.removeprefix("csv_accuracy_results/").removesuffix("results.json")] = json.load(open(file))


for measure in sets["lan_50"]["modifiedopenssl_sha384_0"]:
    for file in sets:
        if "Schmidbauer" not in file:
            for hash in hash_name_map:
                test_ = sets[file][hash][measure]["native"]
                plt.plot(test_["roc"]["fpr"], test_["roc"]["tpr"],
                         label=hash_name_map.get(hash, hash) + ", AUC = " + str(np.round(test_["auc"], 3)))
            plt.plot([0, 1], [0, 1], color="firebrick", linestyle="dotted")
            plt.legend(loc="lower right", prop={'size': 14})
            plt.savefig("csv_accuracy_results/ROC/hash_/" + file
                        + "_" + measure + ".pdf")
            plt.close()

        for hash in sets[file]:
            file_ = file.split("_")[0]
            #if "_" not in file and "all" not in file:
            if "Schmidbauer" not in file:
                for window in [20, 50, 100]:
                    add = "_" + str(window) if window != 100 else ""
                    test_ = sets[file_ + "_" + str(window)][hash][measure]["native"]
                    plt.plot(test_["roc"]["fpr"], test_["roc"]["tpr"],
                             label=str(window) + ", AUC = " + str(np.round(test_["auc"], 3)))
                plt.plot([0, 1], [0, 1], color="firebrick", linestyle="dotted")
                plt.legend(loc="lower right", prop={'size': 14})
                plt.savefig("csv_accuracy_results/ROC/window_/" + hash_name_map.get(hash, hash) + "_" + file_
                            + "_" + measure + ".pdf")
                plt.close()

