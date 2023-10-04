import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from results_graphs_rocs_1 import measure_name_map, hash_name_map

sets = {}
files = glob.glob("csv_accuracy_results/Schmidbauer*.json")
for file in files:
    sets[file.removeprefix("csv_accuracy_results/").removesuffix("results.json")] = json.load(open(file))

reported_AUC = {
    "sha384": 0.414,
    "md5": 0.530,
    "sha3-256": 0.615,
    "sha3-512": 0.879
}

schm_measure_map = {
    "reported": "Threshold detection in [46]",
    #"mean-threshold": "mean",
    "AdaBoost": "Ada",
    "keras": "CNN"
}
def plot_schm(test_name, hash_column = "modified0"):
    columns = {}
    for hash in reported_AUC:
        columns[hash] = {}
        for measure in schm_measure_map:
            if measure == "reported":
                columns[hash][measure] = reported_AUC[hash]
                continue
            if "-threshold" in measure:
                measure_ = measure.removesuffix("-threshold")
                values = sets["Schmidbauer_" + hash + "_50"][hash_column][measure_]["threshold-native"]
            else:
                values = sets["Schmidbauer_" + hash + "_50"][hash_column][measure]["native"]
            columns[hash][measure] = values["auc"]

    r = np.arange(len(reported_AUC))
    i = 0
    _len = 0.7 / len(schm_measure_map)

    colors = ["blue", "orange", "firebrick", "yellow"]

    for measure in schm_measure_map:
        print([columns[c][measure] for c in columns])
        plt.bar(r + i*_len, [columns[c][measure] for c in columns], label=schm_measure_map[measure], color=colors[i], width=_len)
        i+=1

    plt.ylim([0, 1.05])
    plt.ylabel("AUC")
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.axhline(y=0.5)
    plt.xticks(r + 0.3, reported_AUC.keys())

    plt.legend(loc="lower right")
    plt.savefig("csv_accuracy_results/plots/" + test_name +".pdf")
    plt.show()

plot_schm("native")