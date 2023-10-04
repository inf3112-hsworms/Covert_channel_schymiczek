import glob
import json

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


file_name_map = {

}

measure_name_map = {
    "mean": "mean",
    "median": "median",
    "mode": "mode",
    "std": "std",
    "skew": "skew",
    "kurtosis": "kurtosis",
    "hill_est": "Hill est.",
    "pickands_est": "Pickands est.",
    "suessmann": "Süssmann m.",
    "max_hist_inverse": "$\\frac{1}{max(p)}$",
    "log_std": "log. std",
    "FWHM": "FWHM"

}

ML_name_map = {
    "DecisionTree": "DT",
    "GradientBoost": "GB",
    "RandomForest": "RF",
    "AdaBoost": "Ada",
    "linear_SVC": "SVC",
    "LogisticRegression": "LR",
    "RidgeRegressionClassifier": "Ridge",
    "NaiveBayes": "GNB",

}

hash_name_map = {
    "modifiedopenssl_sha256_0": "sha256",
    "modifiedopenssl_md5_0": "md5",
    "modifiedopenssl_sha384_0": "sha384",
    "modifiedopenssl_sha3_256_0": "sha3-256",
    "modifiedopenssl_sha3_512_0": "sha3-512"
}

if __name__=="__main__":
    sets = {}
    files = glob.glob("csv_accuracy_results/all*.json")
    for file in files:
        sets[file.removeprefix("csv_accuracy_results/").removesuffix("results.json")] = json.load(open(file))

    for file in sets:
        for hash in sets[file]:
            for measure in sets[file][hash]:
                for test in sets[file][hash][measure]:
                    test_ = sets[file][hash][measure][test]
                    display = RocCurveDisplay(fpr=test_["roc"]["fpr"], tpr=test_["roc"]["tpr"], roc_auc=test_["auc"],
                    estimator_name = measure_name_map.get(measure, measure))
                    display.plot()
                    plt.savefig("csv_accuracy_results/ROC/"+ hash_name_map.get(hash, hash) + "/" + file
                                + "_" + measure + "_" + test + ".pdf")
                    plt.close()


