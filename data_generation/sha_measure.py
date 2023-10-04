import hashlib
import random
from collections import Counter
from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt
import numpy

alice = open("alice").read()

most_common = [a[0] for a in Counter(alice).most_common()]

timearray = []

for x in range(100000):
    char = random.choice(alice)
    time_ = datetime.now()
    for m in most_common:
        hashlib.sha256(bytes(alice + char, "UTF-8"))
        if m == char:
            break
    time = (datetime.now() - time_).total_seconds()

    timearray.append(time)

hist = numpy.histogram(timearray, bins=1000)[0]
kernel = numpy.ones(10) / 10
smooth = numpy.convolve(hist, kernel, mode="same")
plt.plot(smooth)
plt.yscale("log")

plt.show()
