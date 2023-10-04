import glob

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

from ml_tools.ml_util import read_already_modified_from, get_datasets

hashes = ["sha256_0", "md5_0", "sha384_0", "sha3_256_0"]
get_latex_from_hash_name = lambda hash: hash.replace("_0", "").replace("_", "-")


def suessmann_mannwhitneyu(df, hash, prefix, windows=50):
    number_of_observations = 50

    number_of_window_creations = int(np.ceil(number_of_observations / np.max([np.floor(len(df)/windows), 1]))) + 2

    datasets = get_datasets(df, [windows] * number_of_window_creations)

    ps_1 = []
    for i in range(number_of_observations):
        _, p = mannwhitneyu(datasets[i]["ping time"], datasets[i+1][prefix + hash], alternative="less")
        ps_1.append(p)
    return np.mean(df["ping time"]), ps_1


def plot_mann_whitne(files, filename, hash, prefix="modifiedopenssl_", subscript=""):
    plt.rcParams['text.usetex'] = True
    dfs = read_already_modified_from(files)
    print(filename)
    print(hash)
    getm_w_f = lambda x, y: mannwhitneyu(x, y, alternative="less")[0] / len(x)**2
    MWU = Parallel(n_jobs=7)(delayed(getm_w_f)(df["ping time"], df[prefix + hash]) for df in dfs)

    means = [np.mean(df["ping time"]) for df in dfs]
    plt.scatter(means, MWU, color="green", s=5, marker="o", label="$D$", alpha=0.5)
    plt.xlabel("$\\mu(D)$")
    plt.ylabel("Mann-Whitney U $f$-value")

    plt.savefig("plots/mwu_" + filename + "_" + hash + ".pdf")

    plt.xscale("log")
    plt.savefig("plots/mwu_" + filename + "_" + hash + "_logscale.pdf")
    plt.close()

    color_map={
        20: "yellow",
        50: "goldenrod",
        100: "orange"
    }
    for window in [20, 50, 100]:
        getm_w_f = lambda df: suessmann_mannwhitneyu(df, hash=hash, prefix=prefix, windows=window)
        MWU = Parallel(n_jobs=7)(delayed(getm_w_f)(df) for df in dfs)

        mwu_sum= 0
        mwu_len = 0
        less_than_001=0
        less_than_001_mwu=0
        more_than_001=0
        more_than_001_mwu=0
        to_plot=[]
        for mean, mwu in MWU:
            to_add = np.sum(np.array(mwu) < 0.05)
            mwu_len += len(mwu)
            to_plot.append(to_add / len(mwu))

            mwu_sum += to_add
            if mean < 0.02:
                less_than_001 += len(mwu)
                less_than_001_mwu+= to_add
            else:
                more_than_001 += len(mwu)
                more_than_001_mwu+= to_add
        less_than_ratio = less_than_001_mwu / less_than_001 if less_than_001!=0 else 0
        more_then_ratio = more_than_001_mwu/ more_than_001 if more_than_001!=0 else 0
        print("window: {0} - p-percentage: {1}. Less than 002: {2}, more than 002: {3}".
              format(window, mwu_sum/mwu_len, less_than_ratio, more_then_ratio))
        plt.scatter(means, to_plot, color="orange", s=5, marker="o", label=str(window), alpha=0.5)
        #coeff = np.polyfit(means, to_plot, 3)
        #poly = np.poly1d(coeff)
        #x_p = np.linspace(0, np.max(means))
        #plt.plot(x_p, poly(x_p), color="orange")
        plt.xscale("log")
        plt.xlabel("$\\mu(D)$")
        plt.ylabel("\\% of significant Mann-Whitney U Tests")
        plt.ylim([-0.05, 1.05])
        plt.savefig("plots/mwu_p_"+str(window) + "_" + filename + "_" + hash + ".pdf")
        plt.close()


for hash in hashes:
    plot_mann_whitne(glob.glob("dataframes/*192.168.1.1_*"), "LAN", hash,
                          subscript=get_latex_from_hash_name(hash))
    plot_mann_whitne(glob.glob("dataframes/*wlan__66.211.175.229*"), "medium", hash,
                          subscript=get_latex_from_hash_name(hash))
    plot_mann_whitne(glob.glob("dataframes/*8.8.8.8*"), "google", hash,
                          subscript=get_latex_from_hash_name(hash))
    plot_mann_whitne(glob.glob("dataframes/*australia*"), "aus", hash,
                          subscript=get_latex_from_hash_name(hash))
    plot_mann_whitne(glob.glob("dataframes/*"), "all", hash, subscript=get_latex_from_hash_name(hash))
    plot_mann_whitne(glob.glob("RIPE/modified_data_1/*"), "", hash, prefix="openssl_" ,
                         subscript=get_latex_from_hash_name(hash))

hash = "sha3_512_0"

plot_mann_whitne(glob.glob("dataframes/*192.168.1.1_*"), "LAN", hash,
                      subscript=get_latex_from_hash_name(hash))
plot_mann_whitne(glob.glob("dataframes/*wlan__66.211.175.229*"), "medium", hash,
                      subscript=get_latex_from_hash_name(hash))
plot_mann_whitne(glob.glob("dataframes/*8.8.8.8*"), "google", hash,
                      subscript=get_latex_from_hash_name(hash))
plot_mann_whitne(glob.glob("dataframes/*australia*"), "aus", hash,
                      subscript=get_latex_from_hash_name(hash))
plot_mann_whitne(glob.glob("dataframes/*"), "all", hash,
                     subscript=get_latex_from_hash_name(hash))

hash = "sha512_0"

plot_mann_whitne(glob.glob("RIPE/modified_data_1/*"), "", hash, prefix="openssl_" ,
                     subscript=get_latex_from_hash_name(hash))
