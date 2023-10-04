import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from results_graphs_rocs_1 import measure_name_map, hash_name_map, file_name_map


get_i_map = {
    "lan": 0,
    "wlan": 1,
    "8.8.8.8": 2,
    "81.92.228.153": 3,
    "66.211.175.229": 4,
    "139.130.4.5": 5
}


l = lambda x: [file_name_map.get(t, t) for t in x]


if __name__ == "__main__":

    sets = {}
    files = glob.glob("csv_accuracy_results/Schmidbauer*.json")
    for file in files:
        print(file)
        sets[file.removeprefix("csv_accuracy_results/").removesuffix("results.json")] = json.load(open(file))


    def get_table_row(param):
        p = lambda x: str(np.round(x*100, 1)) + " \\%"
        return "{0} & {1} & {2} ".format(p(param["auc"]), p(param["tpr"]), p(param["tnr"]))


    def make_cnn_table(array):
        return "$n$ & \\multicolumn{{3}}{{c|}}{{ {0} }} \\\\ \n"" \
          {1} \\\\ \n"" \
         20 & {2} \\\\ \n"" \
         50 & {3}  \\\\ \n"" \
         100 & {4}  \\\\ \n".format(
            "} & \multicolumn{3}{c|}{".join(array["top"]),
            "& AUC & TPR  & TNR"*6,
            "&".join(array[20]),
            "&".join(array[50]),
            "&".join(array[100])
        ).replace(" ", "")


    for hash in sets["lan"]:
        table_cnn_absolute = {
            "top": l(get_i_map),
            20: [""] * 6,
            50: [""] * 6,
            100: [""] * 6,
        }

        table_cnn_specific = {
            "top": l(get_i_map),
            20: [""] * 6,
            50: [""] * 6,
            100: [""] *6,
        }
        measure = "keras"
        for file in sets:
            i = get_i_map.get(file, "None")
            if i == "None" and file != "all":
                continue
            hash_ = "modified0" if "Schmidbauer" in file else hash
            if file == "all":
                for test in sets[file][hash_][measure]:
                    if "_" in test and "Schmid" not in test:
                        arr = test.split("_")
                        i = get_i_map[arr[0]]
                        table_cnn_absolute[int(arr[1])][i] = get_table_row(sets[file][hash_][measure][test])
                    elif test in get_i_map.keys():
                        i = get_i_map[test]
                        table_cnn_absolute[100][i] = get_table_row(sets[file][hash_][measure][test])
            elif "_" not in file:
                table_cnn_specific[20][i] = get_table_row(sets[file + "_20"][hash_][measure][file + "_20"])
                table_cnn_specific[50][i] = get_table_row(sets[file + "_50"][hash_][measure][file + "_50"])
                table_cnn_specific[100][i] = get_table_row(sets[file + "_100"][hash_][measure][file])
        if "md5" in hash:
            print(hash)
            print("Absolute:")
            print(table_cnn_absolute)
            print(make_cnn_table(table_cnn_absolute))
            print("Specific: ")
            print(table_cnn_specific)
            print(make_cnn_table(table_cnn_specific))



