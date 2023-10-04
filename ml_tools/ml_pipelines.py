import glob
import hashlib

import numpy as np
from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_tools.ml_util import parallelizable_histogram_creation, \
    ml_train, parallelizable_stats_creation, \
    read_and_modify, get_X_0_X_1, read_already_modified_from


def get_model_ml_histogram_direct(bins=2500):
    model = Sequential()
    # convolutional layer
    model.add(
        Conv1D(50, 2, padding='valid', activation='relu', input_shape=(bins, 1))
    )
    #model.add(MaxPool1D(pool_size=10))
    model.add(Flatten())
    # hidden layer
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(25, activation='sigmoid'))
    # output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=BinaryCrossentropy(),
                  metrics=['accuracy'], optimizer='adam')

    return model


def ml_statistical_measures():
    #return LogisticRegression(n_jobs=7, penalty='l1', solver="saga")
    return DecisionTreeClassifier(max_depth=10, class_weight={0:1.5, 1:1})


def get_gradient_boost():
    return GradientBoostingClassifier(max_depth=5, n_iter_no_change=3, n_estimators=20)

def get_random_forest():
    return RandomForestClassifier(n_jobs=7, n_estimators=20, max_depth=5, class_weight={0:1, 1:1.5})


def test_with_data(model, file, hashfunc, name="", windows=[100], keras=False):
    testdata = read_and_modify(file, hashfunc)
    stat_data = get_X_0_X_1(testdata, windows, parallelizable_stats_creation if not keras else parallelizable_histogram_creation)
    X_0 = [a[0] for a in stat_data]
    X_1 = [a[1] for a in stat_data]
    none_filter = lambda x: x is not None
    X_0 = list(filter(none_filter, X_0))
    X_1 = list(filter(none_filter, X_1))
    _, X, _, y = train_test_split(X_0 + X_1, [0]*len(X_0) + [1]*len(X_1), train_size=1)
    if not keras:
        y_pred = model.predict(X)
        print("accuracy {1}: {0}".format(model.score(X, y), name))
        print(confusion_matrix(y, y_pred, normalize="all"))
    else:
        X = np.array([x.todense() for x in X])
        score, acc = model.evaluate(X, np.array(y))
        y_pred = model.predict(X)
        y_pred = [np.round(y_) for y_ in y_pred]
        print(confusion_matrix(y, y_pred, normalize="all"))
        print("accuracy {0}: {1} (score: {2})".format(name, acc, score))

if __name__ == "__main__":

    hashfunc = hashlib.sha256
    data_frames = read_already_modified_from(glob.glob("RIPE/modified_data_1/*"))
    #ml_train(data_frames, get_model_ml_histogram_direct(),
    #         parallelizable_histogram_creation, "models/nn_direct_", epochs=15)

    p=lambda x: parallelizable_stats_creation(x, prefix="openssl_sha256_")



    model1 = ml_train(data_frames, ml_statistical_measures(),
             p, "models/random_forest_stats_direct_", keras=False,
             windows=[100, 500])

    test_with_data(model1, "../diagnostics/dia_medium_wlan__81.92.228.153_2022_05_07_01_45.csv", hashfunc, "medium latency")
    test_with_data(model1, "../diagnostics/dia_australia__139.130.4.5_2022_05_05_22_01.csv", hashfunc, "high latency")
    test_with_data(model1, "../diagnostics/dia_long_time_lan__192.168.1.1_2022_05_06_18_07.csv", hashfunc, "low latency")

    #print(export_text(model1, feature_names=f_names))

    model2 = ml_train(data_frames, get_gradient_boost(),
             p, "models/random_forest_stats_direct_", keras=False,
             windows=[100, 500])

    test_with_data(model2, "../diagnostics/dia_medium_wlan__81.92.228.153_2022_05_07_01_45.csv", hashfunc, "medium latency")
    test_with_data(model2, "../diagnostics/dia_australia__139.130.4.5_2022_05_05_22_01.csv", hashfunc, "high latency")
    test_with_data(model2, "../diagnostics/dia_long_time_lan__192.168.1.1_2022_05_06_18_07.csv", hashfunc, "low latency")


    model4 = ml_train(data_frames, get_random_forest(),
             p, "models/random_forest_stats_direct_", keras=False,
             windows=[100, 500])

    test_with_data(model4, "../diagnostics/dia_medium_wlan__81.92.228.153_2022_05_07_01_45.csv", hashfunc, "medium latency")
    test_with_data(model4, "../diagnostics/dia_australia__139.130.4.5_2022_05_05_22_01.csv", hashfunc, "high latency")
    test_with_data(model4, "../diagnostics/dia_long_time_lan__192.168.1.1_2022_05_06_18_07.csv", hashfunc, "low latency")


    def get_SVC():
        return SVC()


    model5 = ml_train(data_frames, get_SVC(),
             p, "models/random_forest_stats_direct_", keras=False,
             windows=[100, 500])

    test_with_data(model5, "../diagnostics/dia_medium_wlan__81.92.228.153_2022_05_07_01_45.csv", hashfunc, "medium latency")
    test_with_data(model5, "../diagnostics/dia_australia__139.130.4.5_2022_05_05_22_01.csv", hashfunc, "high latency")
    test_with_data(model5, "../diagnostics/dia_long_time_lan__192.168.1.1_2022_05_06_18_07.csv", hashfunc, "low latency")



    model3 = ml_train(data_frames, get_model_ml_histogram_direct(),
              lambda x: parallelizable_histogram_creation(x, prefix="openssl_sha256_"), "models/random_RF_comparison_", sparse=True, keras=True,
              windows=[100, 500], epochs=20)
    #
    test_with_data(model3, "../diagnostics/dia_medium_wlan__81.92.228.153_2022_05_07_01_45.csv", hashfunc, "medium latency", keras=True)
    test_with_data(model3, "../diagnostics/dia_australia__139.130.4.5_2022_05_05_22_01.csv", hashfunc, "high latency", keras=True)
    test_with_data(model3, "../diagnostics/dia_long_time_lan__192.168.1.1_2022_05_06_18_07.csv", hashfunc, "low latency", keras=True)
