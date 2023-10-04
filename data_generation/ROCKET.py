import glob

import numpy.fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ml_tools.ml_util import read_already_modified_from, get_X_0_X_1, parallelizable_cutoff
from pyts.transformation import ROCKET

data_frames = read_already_modified_from(glob.glob("dataframes/*66.211.175*"))
X_0, X_1 = [], []
for df in data_frames:
    X_0_X_1 = get_X_0_X_1(df, [250], lambda x: parallelizable_cutoff(x, prefix="modifiedopenssl_sha256_", window=250), shuffle=False)
    for _X_0, _X_1 in X_0_X_1:
        if _X_0 is not None and _X_1 is not None:
            X_0.append(_X_0)
            X_1.append(_X_1)
    del df
del data_frames

print("starting rocket")
rocket = ROCKET(n_kernels=500)
X = X_0 + X_1
#X.reshape(-1, 1)
print("starting rocket training")

y = numpy.array([0]*len(X_0) + [1] * len(X_1))

model = RidgeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y)

rocket.fit(X_train)
model.fit(rocket.transform(X_train), y_train)
print("accuracy training: " + str(model.score(rocket.transform(X_train), y_train)))

X_test_t = rocket.transform(X_test)
print(model.score(X_test_t, y_test))
y_pred = model.predict(X_test_t)
print(confusion_matrix(y_test, y_pred, normalize="all"))

print("-------------------------FFT-------------------------")

for x in X_0:
    x = numpy.fft.fft(x)

for x in X_1:
    x = numpy.fft.fft(x)

X = X_0 + X_1
y = numpy.array([0]*len(X_0) + [1] * len(X_1))

model = RandomForestClassifier(n_jobs=7, n_estimators=100, max_depth=7)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)
print("accuracy training: " + str(model.score(X_train, y_train)))

y_pred = model.predict(X_test)
print("accuracy {1}: {0}".format(model.score(X_test, y_test), "FFT rf"))
print(confusion_matrix(y_test, y_pred, normalize="all"))

print("---------------------FFT2--------------------------")

model = DecisionTreeClassifier(max_depth=7)

model.fit(X_train, y_train)
print("accuracy training: " + str(model.score(X_train, y_train)))

y_pred = model.predict(X_test)
print("accuracy {1}: {0}".format(model.score(X_test, y_test), "FFT rf"))
print(confusion_matrix(y_test, y_pred, normalize="all"))
print(export_text(model))
