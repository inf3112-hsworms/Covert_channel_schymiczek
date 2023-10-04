import glob

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ml_tools.ml_util import read_already_modified_from

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

hist = lambda x: np.histogram(x, bins=250, range=[0, 0.03])

LAN = read_already_modified_from(glob.glob("dataframes/*lan__192.168.1.115*"))
WLAN = read_already_modified_from(glob.glob("dataframes/*wlan__192.168.1.115*"))
GOOGLE = read_already_modified_from(glob.glob("dataframes/*wlan_2__8.8.8.8*"))
MEDIUM = read_already_modified_from(glob.glob("dataframes/*wlan__66.211.175.229*"))
MEDIUM2 = read_already_modified_from(glob.glob("dataframes/*wlan__81.92.228.153*"))
AUS = read_already_modified_from(glob.glob("dataframes/*australia*"))

def compare_min_max(set, do_compare = True,
                    columns=["ping time", "openssl_md5_0", "openssl_sha3_512_0"],
                    hist_func=hist, filename="test", combination=[0, 1, 2, 3, 4, 5],
                    title_1="", title_2="", labels=["$D$", "$D + \gamma_{md5}$", "$D + \gamma_{sha3-512}$"],
                    colors_light=["cyan", "tomato", "gold"],
                    colors_dark=["blue", "firebrick", "orange"]):
    if do_compare:
        index_min = min(range(len(set)), key=[np.median(s[columns[0]]) for s in set].__getitem__)

        index_max = max(range(len(set)), key=[np.median(s[columns[0]]) for s in set].__getitem__)
    else:
        index_min=0

    l = []
    for c in columns:
        l.append(set[index_min][c])

    if do_compare:
        for c in columns:
            l.append(set[index_max][c])

    h = []

    for l_ in l:
        h_, scale = hist_func(l_)
        h.append(np.array(h_) / len(l_))
    h_max_1 = np.max(h[0])
    if do_compare:
        h_max_2 = np.max(h[len(columns)])
    scale = scale[:-1]
    plt.rcParams['text.usetex'] = True
    for i in range(len(columns)):
        plt.plot(scale, h[combination[i]], c=colors_light[i], label=labels[i])
        print()
        print(labels[i])
        print(filename)
        print("mean: " + str(np.median(l[i])))
        plt.axvline(x=np.median(l[i]), color=colors_light[i], alpha=0.5)

    plt.title(title_1)
    plt.xlabel("seconds")
    plt.ylim((0, 1.1*h_max_1))
    plt.legend(loc="upper right", prop={'size': 16})
    plt.savefig("plots/" + filename + "_light.pdf")
    plt.show()

    if do_compare:

        for i in range(len(columns)):
            plt.plot(scale, h[combination[len(columns) + i]], c=colors_dark[i], label=labels[i])
            print()
            print(labels[i])
            print(filename)
            print("mean: " + str(np.median(l[len(columns) + i])))
            plt.axvline(x=np.median(l[len(columns) + i]), color=colors_dark[i], alpha=0.5)
        plt.title(title_2)
        plt.ylim((0, 1.1*h_max_1))
        plt.xlabel("seconds")
        plt.legend(loc="upper right", prop={'size': 16})
        plt.savefig("plots/" + filename + "_dark.pdf")

        plt.show()


if __name__ == "__main__":
    c_2 = ["ping time", "modifiedopenssl_md5_0", "modifiedopenssl_sha3_512_0"]

    compare_min_max(LAN, columns= c_2, filename="_LAN_complete")
    compare_min_max(WLAN, columns= c_2, filename="_WLAN_complete")
    compare_min_max(GOOGLE, columns= c_2,
                    filename="_GOOGLE_complete",
                    hist_func=lambda x: np.histogram(x, bins=250, range=[0.005, 0.03]))
    compare_min_max(MEDIUM, columns=c_2,
                    filename="_MEDIUM_complete",
                    hist_func=lambda x: np.histogram(x, bins=250, range=[0.15, 0.35]))
    compare_min_max(MEDIUM2, columns=c_2,
                    filename="_MEDIUM2_complete",
                    hist_func=lambda x: np.histogram(x, bins=250, range=[0, 0.05]))
    compare_min_max(AUS, columns=c_2,
                    filename="_AUS_complete",
                    hist_func=lambda x: np.histogram(x, bins=250, range=[0.3, 0.4]))
