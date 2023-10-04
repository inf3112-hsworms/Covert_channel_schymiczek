import gc
import glob
import itertools
import json

import numpy as np
import pandas
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Lasso
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml_tools.ml_pipelines import get_model_ml_histogram_direct
from ml_tools.ml_util import read_already_modified_from, get_datasets, parallelizable_stats_creation, \
    parallelizable_histogram_creation

def ml_algs():
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=10),
        "GradientBoost": GradientBoostingClassifier(max_depth=5, n_iter_no_change=3, n_estimators=20),
        "RandomForest": RandomForestClassifier(n_jobs=7, n_estimators=20, max_depth=5),
        "linear_SVC": LinearSVC(),
        "LogisticRegression": LogisticRegression(),
        "RidgeRegressionClassifier": RidgeClassifier(),
        "NaiveBayes": GaussianNB(),
        "Lasso": Lasso(),
        "AdaBoost": AdaBoostClassifier()
    }


ACC = "acc"
ROC = "roc"
AUC = "auc"
TPR = "tpr"
TNR = "tnr"


def add_accuracy_score(A, y_test, y_pred):
    A[ACC] = accuracy_score(y_test, np.round(y_pred))
    print(float(A[ACC]))
    roc = roc_curve(y_test, y_pred)
    A[ROC] = {}
    A[AUC] = float(auc(roc[0], roc[1]))
    A[ROC]["fpr"] = [float(r) for r in list(roc[0])]
    A[ROC]["tpr"] = [float(r) for r in list(roc[1])]
    A[ROC]["threshold"] = [float(r) for r in list(roc[2])]
    conf = confusion_matrix(y_test, np.round(y_pred), normalize="true").ravel()
    A[TPR] = conf[3]
    A[TNR] = conf[0]


def get_Xy(datasets, column):
    stat_X_0_X_1 = Parallel(n_jobs=7)(delayed(lambda x: parallelizable_stats_creation(x, prefix=column, choose=True))
                                      (df) for df in datasets)
    X_0, X_1 = [], []
    for x_0, x_1 in stat_X_0_X_1:
        if x_0 is not None and x_1 is not None:
            X_0.append(x_0)
            X_1.append(x_1)
    y_0 = len(X_0)
    y_1 = len(X_1)
    print(f"lens: {y_0}, {y_1}")
    X = list(itertools.chain(X_0, X_1))
    y = np.concatenate((np.array([0] * y_0), np.array([1] * y_1)))
    return X, y


def get_test_train_stat(datasets, column):
    X, y = get_Xy(datasets, column)
    return train_test_split(X, y, train_size=0.75)


def get_Xy_keras(datasets, column):
    stat_X_0_X_1 = Parallel(n_jobs=7)(delayed(lambda x: parallelizable_histogram_creation(x, prefix=column, choose=True))
                                      (df) for df in datasets)
    X_0, X_1 = [], []
    for x_0, x_1 in stat_X_0_X_1:
        if x_0 is not None and x_1 is not None:
            X_0.append(x_0)
            X_1.append(x_1)
    y_0 = len(X_0)
    y_1 = len(X_1)
    print(f"lens: {y_0}, {y_1}")
    X = list(itertools.chain(X_0, X_1))
    y = np.concatenate((np.array([0] * y_0), np.array([1] * y_1)))
    return X, y


def get_test_train_keras(datasets, column):
    X, y = get_Xy_keras(datasets, column)
    return train_test_split(X, y, train_size=0.75)


def column_logic(param, c):
    if param == "same" and c != "modified0":
        return c
    elif param == "same":
        return "modifiedopenssl_md5_0"
    elif param == "all": # possible future feature
        return c
    else:
        return param


p = lambda x: [x_[1] for x_ in x]


def create_accuracy_summary_for_all_files(files, testfiles, windows):
    dfs = read_already_modified_from(files)
    datasets = Parallel(n_jobs=7)(delayed(get_datasets)(df, windows) for df in dfs)
    all_datasets = list(itertools.chain(*datasets))
    return create_accuracy_summary(all_datasets, pandas.concat(dfs), testfiles)

def create_accuracy_summary(all_datasets, all_datapoints, testfiles):

    accuracy_json = {}

    print("--- loading training dataset ---")
    columns = all_datasets[0].columns

    print("--- loading test datasets ---")
    test_datasets = {}
    for test in testfiles:
        print(test)
        windows = testfiles[test][2]
        test_files = testfiles[test][0]
        if "schmid" not in test_files:
            test_files = glob.glob(test_files)
            test_dfs = read_already_modified_from(test_files)
            test_ds = Parallel(n_jobs=7)(delayed(get_datasets)(df, windows) for df in test_dfs)
            all_ds = list(itertools.chain(*test_ds))
            test_datasets[test] = [all_ds, testfiles[test][1]]
        else:
            arr = files.split("_")
            if len(arr) == 2:
                schm0 = glob.glob("data/" + arr[1] + "/best/00_0/*.csv")
                schm1 = glob.glob("data/" + arr[1] + "/best/02_100/*.csv")
            else:
                schm0 = glob.glob("data/00_0/*.csv")
                schm1 = glob.glob("data/02_100/*.csv")
            df_list = []
            for s0, s1 in zip(schm0, schm1):
                new_df = DataFrame()
                df0 = pandas.read_csv(s0)
                df1 = pandas.read_csv(s1)
                new_df["ping time"] = df0["Time since previous frame in this TCP stream"]
                new_df["modified"] = df1["Time since previous frame in this TCP stream"]
                df_list.append(new_df)
            test_ds = Parallel(n_jobs=7)(delayed(get_datasets)(df, windows) for df in df_list)
            all_ds = list(itertools.chain(*test_ds))
            test_datasets[test] = [all_ds, testfiles[test][1]]


    means = {}
    for c in columns:
        if c == "ping time" or "modified" in c and c.endswith("0"):
            means[c] = np.mean(all_datapoints[c])

    del all_datapoints

    for c in columns:
        gc.collect()
        if "modified" in c and c.endswith("0") and "sha512" not in c:
            print("----------------------------------")
            print(c)
            print("----------------------------------")
            print()
            print("\t--- generating statistical data from main dataset ---")
            X_train, X_test, y_train, y_test = get_test_train_stat(all_datasets, c)
            test_dataset_stat = {}
            print("\t --- generating statistical data from test dataset ---")
            for test in test_datasets:
                print("\t\t" + test)
                test_dataset_stat[test] = get_Xy(test_datasets[test][0], column_logic(test_datasets[test][1], c))
            accuracy_json[c] = {}

            #
            # One measure
            #
            for measure in X_test[0].index:
                accuracy_json[c][measure] = {}
                print()
                print(measure + ": ")
                print()
                X_threshold = np.array([x[measure] for x in X_train])
                X_threshold_test = np.array([x[measure] for x in X_test])
                if measure in ["mean"]:
                    mean_0 = means["ping time"]
                    mean_1 = means[c]
                    threshold = (mean_1 + mean_0) / 2
                    reverse = False
                else:
                    mean_0 = np.mean(X_threshold[y_train == 0])
                    mean_1 = np.mean(X_threshold[y_train == 1])
                    threshold = (mean_1 + mean_0)/2
                    reverse = mean_1 < mean_0
                y_pred = (X_threshold_test > threshold) if not reverse else (X_threshold_test < threshold)
                print("threshold-native")
                accuracy_json[c][measure]["threshold-native"] = {}
                add_accuracy_score(accuracy_json[c][measure]["threshold-native"], y_test, y_pred)
                for test in test_dataset_stat:
                    print("threshold-" + test)
                    X = test_dataset_stat[test][0]
                    accuracy_json[c][measure]["threshold-" + test] = {}
                    X_test_threshold = np.array([x[measure] for x in X])
                    add_accuracy_score(accuracy_json[c][measure]["threshold-" + test], test_dataset_stat[test][1],
                                       X_test_threshold > threshold if not reverse else (X_test_threshold < threshold))

                X_train_spec = [[x["mean"], x[measure]] for x in X_train]
                X_test_spec = [[x["mean"], x[measure]] for x in X_test]
                model = LogisticRegression()
                model.fit(X_train_spec, y_train)
                print(model.coef_)
                y_pred = p(model.predict_proba(X_test_spec))
                y_pred_train = p(model.predict_proba(X_train_spec))
                print("native")
                accuracy_json[c][measure]["native"] = {}
                add_accuracy_score(accuracy_json[c][measure]["native"], y_test, y_pred)
                print("training")
                accuracy_json[c][measure]["training"] = {}
                add_accuracy_score(accuracy_json[c][measure]["training"], y_train, y_pred_train)
                for test in test_dataset_stat:
                    print(test)
                    X = test_dataset_stat[test][0]
                    accuracy_json[c][measure][test] = {}
                    add_accuracy_score(accuracy_json[c][measure][test], test_dataset_stat[test][1], p(model.predict_proba([[x["mean"], x[measure]] for x in X])))

            #
            # ML Algorithms
            #
            ml_alg = ml_algs()
            for alg in ml_alg:
                accuracy_json[c][alg] = {}
                model = ml_alg[alg]
                model.fit(X_train, y_train)
                print(alg)
                try:
                    y_pred = p(model.predict_proba(X_test))
                    y_pred_train = p(model.predict_proba(X_train))
                except:
                    y_pred = model.predict(X_test)
                    y_pred_train = model.predict(X_train)
                print("native")
                accuracy_json[c][alg]["native"] = {}
                add_accuracy_score(accuracy_json[c][alg]["native"], y_test, y_pred)
                print("training")
                accuracy_json[c][alg]["training"] = {}
                add_accuracy_score(accuracy_json[c][alg]["training"], y_train, y_pred_train)
                for test in test_dataset_stat:
                    print(test)
                    accuracy_json[c][alg][test] = {}
                    try:
                        add_accuracy_score(accuracy_json[c][alg][test], test_dataset_stat[test][1], p(model.predict_proba(test_dataset_stat[test][0])))
                    except:
                        add_accuracy_score(accuracy_json[c][alg][test], test_dataset_stat[test][1],
                               model.predict(test_dataset_stat[test][0]))

            #
            # Keras
            #
            accuracy_json[c]["keras"] = {}
            model = get_model_ml_histogram_direct()
            X_train, X_test, y_train, y_test = get_test_train_keras(all_datasets, c)
            X_train = np.array([x.todense() for x in X_train])
            X_test = np.array([x.todense() for x in X_test])
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

            print("keras: ")
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            del X_test
            del X_train
            print("native")
            accuracy_json[c]["keras"]["native"] = {}
            add_accuracy_score(accuracy_json[c]["keras"]["native"], y_test, y_pred)
            print("training")
            accuracy_json[c]["keras"]["training"] = {}
            add_accuracy_score(accuracy_json[c]["keras"]["training"], y_train, y_pred_train)
            for test in test_datasets:
                X, y = get_Xy_keras(test_datasets[test][0], column_logic(test_datasets[test][1], c))
                print(test)
                X = np.array([x.todense() for x in X])
                accuracy_json[c]["keras"][test] = {}
                add_accuracy_score(accuracy_json[c]["keras"][test], y, model.predict(X))
                del X


    #
    # Keras all hash
    #

    return accuracy_json


testfiles_2 = { # name: [files, hash, windows]
    "Schmidbauer_sha3-256": ["schmid_sha3-256", "modified", [100]],
    "Schmidbauer_sha3-512": ["schmid_sha3-512", "modified", [100]],
    "Schmidbauer_sha384": ["schmid_sha384", "modified", [100]],
    "Schmidbauer_md5": ["schmid_md5", "modified", [100]],

    "Schmidbauer_sha3-256-20": ["schmid_sha3-256", "modified", [20]],
    "Schmidbauer_sha3-512-20": ["schmid_sha3-512", "modified", [20]],
    "Schmidbauer_sha384-20": ["schmid_sha384", "modified", [20]],
    "Schmidbauer_md5-20": ["schmid_md5", "modified", [20]],

    "Schmidbauer_sha3-256_50": ["schmid_sha3-256", "modified", [50]],
    "Schmidbauer_sha3-512_50": ["schmid_sha3-512", "modified", [50]],
    "Schmidbauer_sha384_50": ["schmid_sha384", "modified", [50]],
    "Schmidbauer_md5_50": ["schmid_md5", "modified", [50]],
}

testfiles = { # name: [files, hash, windows]
    "lan": ["dataframes/*lan__192.168.1.115*", "same", [100]],
    "wlan": ["dataframes/*wlan__192.168.1.115*", "same", [100]],
    "66.211.175.229": ["dataframes/*66.211.175.229*", "same", [100]],
    "81.92.228.153": ["dataframes/*81.92.228.153*", "same", [100]],
    "139.130.4.5": ["dataframes/*139.130.4.5*", "same", [100]],
    "8.8.8.8": ["dataframes/*8.8.8.8*", "same", [100]],

    "lan_20": ["dataframes/*lan__192.168.1.115*", "same", [20]],
    "wlan_20": ["dataframes/*wlan__192.168.1.115*", "same", [20]],
    "66.211.175.229_20": ["dataframes/*66.211.175.229*", "same", [20]],
    "81.92.228.153_20": ["dataframes/*81.92.228.153*", "same", [20]],
    "139.130.4.5_20": ["dataframes/*139.130.4.5*", "same", [20]],
    "8.8.8.8_20": ["dataframes/*8.8.8.8*", "same", [20]],

    "lan_50": ["dataframes/*lan__192.168.1.115*", "same", [50]],
    "wlan_50": ["dataframes/*wlan__192.168.1.115*", "same", [50]],
    "66.211.175.229_50": ["dataframes/*66.211.175.229*", "same", [50]],
    "81.92.228.153_50": ["dataframes/*81.92.228.153*", "same", [50]],
    "139.130.4.5_50": ["dataframes/*139.130.4.5*", "same", [50]],
    "8.8.8.8_50": ["dataframes/*8.8.8.8*", "same", [50]],
}

file_lists = {
    "139.130.4.5": "dataframes/*139.130.4.5*",
    "8.8.8.8": "dataframes/*8.8.8.8*",
    "lan": "dataframes/*lan__192.168.1.115*",
    "wlan": "dataframes/*wlan__192.168.1.115*",
    "66.211.175.229": "dataframes/*66.211.175.229*",
    "81.92.228.153": "dataframes/*81.92.228.153*",
}

file_lists_schmidbauer = {
    "Schmidbauer": "schmidbauer",
    "Schmidbauer_sha3-256": "schmidbauer_sha3-256",
    "Schmidbauer_sha3-512": "schmidbauer_sha3-512",
    "Schmidbauer_sha384": "schmidbauer_sha384",
    "Schmidbauer_md5": "schmidbauer_md5",
}




for files in file_lists:
    print("##################################")
    print(files)
    print("##################################")
    print()
    gc.collect()
    if "Schmidbauer" in files:
        arr = files.split("_")
        if len(arr)==2:
            schm0 = glob.glob("data/"+arr[1]+"/best/00_0/*.csv")
            schm1 = glob.glob("data/"+arr[1]+"/best/02_100/*.csv")
        else:
            schm0 = glob.glob("data/00_0/*.csv")
            schm1 = glob.glob("data/02_100/*.csv")
        df_list = []
        for s0, s1 in zip(schm0, schm1):
            new_df = DataFrame()
            df0 = pandas.read_csv(s0)
            df1 = pandas.read_csv(s1)
            len_ = min(len(df0), len(df1)) - 1
            new_df["ping time"] = df0.iloc[0:len_]["Time since previous frame in this TCP stream"]
            new_df["modified0"] = df1[0:len_]["Time since previous frame in this TCP stream"]
            df_list.append(new_df)
        ds = Parallel(n_jobs=7)(delayed(get_datasets)(df, [20, 50, 100] * 2) for df in df_list)
        all_ds = list(itertools.chain(*ds))
        res_series = create_accuracy_summary(all_ds, pandas.concat(df_list), testfiles)
    else:
        res_series = create_accuracy_summary_for_all_files(glob.glob(file_lists[files]), testfiles, [20, 50, 100] * 4)
    json_string = json.dumps(res_series, indent = 4)
    myfile = open("csv_accuracy_results/"+ files +"results.json", "w")
    myfile.write(json_string)
    myfile.close()

for files in file_lists:
    print("##################################")
    print(files)
    print("##################################")
    print()
    for window in [50, 20, 100]:
        gc.collect()
        if "Schmidbauer" in files:
            arr = files.split("_")
            if len(arr) == 2:
                schm0 = glob.glob("data/" + arr[1] + "/best/00_0/*.csv")
                schm1 = glob.glob("data/" + arr[1] + "/best/02_100/*.csv")
            else:
                schm0 = glob.glob("data/00_0/*.csv")
                schm1 = glob.glob("data/02_100/*.csv")
            df_list = []
            for s0, s1 in zip(schm0, schm1):
                new_df = DataFrame()
                df0 = pandas.read_csv(s0)
                df1 = pandas.read_csv(s1)
                len_ = min(len(df0), len(df1)) - 1
                new_df["ping time"] = df0.iloc[0:len_]["Time since previous frame in this TCP stream"]
                new_df["modified0"] = df1[0:len_]["Time since previous frame in this TCP stream"]
                df_list.append(new_df)
            ds = Parallel(n_jobs=7)(delayed(get_datasets)(df, [window]*5) for df in df_list)
            all_ds = list(itertools.chain(*ds))
            res_series = create_accuracy_summary(all_ds, pandas.concat(df_list), testfiles)
        else:
            res_series = create_accuracy_summary_for_all_files(glob.glob(file_lists[files]), testfiles, [window] * 4)
        json_string = json.dumps(res_series, indent = 4)
        myfile = open("csv_accuracy_results/"+ files + "_" + str(window) + "results.json", "w")
        myfile.write(json_string)
        myfile.close()
