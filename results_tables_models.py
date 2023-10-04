import glob
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

from results_graphs_rocs_1 import measure_name_map, hash_name_map, file_name_map, ML_name_map
from results_tables_CNN import get_i_map, l

sets = {}
files = glob.glob("csv_accuracy_results/*_50*results.json") + glob.glob("csv_accuracy_results/*all*results.json")
for file in files:
    print(file)
    sets[file.removeprefix("csv_accuracy_results/").removesuffix("results.json")] = json.load(open(file))


def get_table_row(param, acc=False):
    p = lambda x: str(np.round(x*100, 1)) + " "
    return "{0} & {1} & {2} ".format(p(param["auc"] if not acc else param["acc"]), p(param["tpr"]), p(param["tnr"]))


def make_cnn_table(table, ml=False, acc=False):
    desc = "& AUC & TPR  & TNR" if not acc else "& Acc & TPR  & TNR"
    ret = ("measure" if not ml else "alg.") + " & \\multicolumn{3}{c|}{ " +"} & \multicolumn{3}{c|}{".join(table["top"])+  "} \\\\ \n" \
                          + desc*6 + "\\\\ \n"
    measure_map = measure_name_map if not ml else ML_name_map
    for measure in measure_map:
        ret += measure_map.get(measure) + "&" + "&".join(table[measure]) + " \\\\ \n"

    return ret


for hash in sets["lan_50"]:
    table_threshold = {
        "top": l(get_i_map)
    }

    table_log_reg = {
        "top": l(get_i_map)
    }

    table_ml = {
        "top": l(get_i_map)
    }

    table_ml_general = {
        "top": l(get_i_map)
    }
    for file in sets:
        i = get_i_map.get(file.removesuffix("_50"), "None")
        if i=="None" or file == "all":
            continue
        for measure in measure_name_map:
            hash_ = "modified0" if "Schmidbauer" in file else hash
            if "_50" in file:
                table_threshold[measure] = table_threshold.get(measure, [""]*6)
                table_log_reg[measure] = table_log_reg.get(measure, [""]*6)

                table_threshold[measure][i] = get_table_row(sets[file][hash_][measure]["native"], acc=True)
                table_log_reg[measure][i] = get_table_row(sets[file][hash_][measure]["native"])
        for measure in ML_name_map:
            hash_ = "modified0" if "Schmidbauer" in file else hash
            if "_50" in file:
                table_ml[measure] = table_ml.get(measure, [""]*6)
                table_ml[measure][i] = get_table_row(sets[file][hash_][measure]["native"])

    for measure in ML_name_map:
        table_ml_general[measure] = table_ml_general.get(measure, [""] * 6)
        for test in sets["all"][hash][measure]:
            if "_50" in test and "Schmid" not in test:
                hash_ = "modified0" if "Schmidbauer" in file else hash
                arr = test.split("_")
                i = get_i_map[arr[0]]
                table_ml_general[measure][i] = get_table_row(sets["all"][hash_][measure][test])
    if "md5" in hash:
        print(hash)
        print("Threshold:")
        print(table_threshold)
        print(make_cnn_table(table_threshold, acc=True))
        print("ML: ")
        print(table_ml)
        print(make_cnn_table(table_ml, ml=True))
        print("ML-Gen: ")
        print(table_ml_general)
        print(make_cnn_table(table_ml_general, ml=True))


