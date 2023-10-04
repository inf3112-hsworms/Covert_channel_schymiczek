from collections import Counter
from copy import copy, deepcopy
from datetime import datetime
import random

import matplotlib.pyplot as plt
import numpy
import numpy as np
import hashlib

import scipy.stats
from pythonping import ping


def smoothness(hist):
    n = len(hist)
    res = 0
    for i in range(n - 1):
        res += numpy.abs(hist[i + 1] - hist[i])
    return 1 / res if res != 0 else 0


def obesity_index(data, i_=4):
    new_data = deepcopy(data)
    n = len(data)
    res = 0
    num = 0
    if i_ == 3:
         return 0
    elif i_==4:
        for _ in range(20):
            random.shuffle(new_data)
            for i in range(n - 4):
                window = data[i:i + 4]
                sorted = numpy.sort(window)
                num+=1
                if sorted[3] > sorted[2] + sorted[1] - sorted[0]:
                    res += 1
        return res / num if res != 0 else 0
    else:
        raise ValueError()


def hill_estimator(k, sorted):
    n = len(sorted)
    if k < n:
        subtract = numpy.log(sorted[n - k - 1])
        res = 0

        for i in range(k):
            val = numpy.log(sorted[n - i - 1])
            res += val - subtract

        return k * 1 / res if res != 0 else 0
    else:
        return -1


def plot(toplot, title="", min=0, max=0.02, bins=500):
    hist_, scale = np.histogram(toplot, bins=bins, range=[min, max])
    hist = np.array(hist_) / len(toplot)
    plt.plot(scale[0:-1], hist)
    plt.xlabel("Seconds")
    plt.ylabel("Proportional Frequency")
    plt.title(title)
    plt.show()

    survival_function = copy(hist)
    for i in range(len(survival_function)):
        if i - 1 >= 0:
            survival_function[i] = survival_function[i] + survival_function[i - 1]
    survival_function = numpy.array([1] * len(survival_function)) - numpy.array(survival_function)

    plt.plot(scale[0:-1], survival_function)
    plt.xlabel("Seconds")
    plt.ylabel("Proportion Surviving")
    plt.title(title)
    plt.show()

    sorted = numpy.sort(toplot)
    n = len(sorted)
    print("({0}, \t hill estimator {1} at 0.01,\t {2} at 0.05,\t {3} at 0.1)".format(title,
                                                                                     hill_estimator(int(0.01 * n),
                                                                                                    sorted),
                                                                                     hill_estimator(int(0.05 * n),
                                                                                                    sorted),
                                                                                     hill_estimator(int(0.1 * n),
                                                                                                    sorted)))

    print("{0} \t obesity index: {1},\t tail obesity index: {2},\t extreme tail obesity index: {3}".format(
        title, obesity_index(toplot), obesity_index(sorted[int(0.75 * n):]), obesity_index(sorted[int(0.95 * n):-1])
    ))
    print("{0}\t moments: 1: {1},\t 2: {2},\t 3: {3},\t 4: {4}".format(
        title, numpy.mean(toplot), numpy.var(toplot), scipy.stats.skew(toplot), scipy.stats.kurtosis(toplot)
    ))
    print("{0}\t smoothness: {1}".format(
        title, smoothness(numpy.convolve(hist, [1, 1, 1, 1, 1]))
    ))
    print("--------------------------------------------------------------")


def add_cc_noise(data, hashfunc=hashlib.sha256):
    newres = []
    alice = open("alice").read()
    most_common = [a[0] for a in Counter(alice).most_common()]
    channel = []
    for r in data:
        char = random.choice(alice)
        time_ = datetime.now()
        for m in most_common:
            hashfunc(bytes(alice + char + str(time_), "UTF-8"))
            if m == char:
                break
        time = (datetime.now() - time_).total_seconds()
        newres.append(r + time)
        channel.append(time)

    return newres, channel
