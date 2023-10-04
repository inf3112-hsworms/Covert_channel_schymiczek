import glob

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from ml_tools.ml_util import read_already_modified_from, create_statistical_data

hashes = ["sha256_0", "md5_0", "sha384_0", "sha3_256_0"]
get_latex_from_hash_name = lambda hash: hash.replace("_0", "").replace("_", "-")


def plot_stat(stats, name, filename, do_log=False, desc="", title="", scale="log", subscript="", **kwargs):
    plt.rcParams['text.usetex'] = True


    x = [stat[0]["mean"] for stat in stats]
    y_0 = [stat[0][name] for stat in stats]
    y_1 = [stat[1][name] for stat in stats]
    plt.scatter(x, y_0, color="cyan", s=5, marker="o", label="$D$", alpha=0.5, **kwargs)
    plt.scatter(x, y_1, color="tomato", s=5, marker="o", label="$D+\\gamma_{"+subscript+"}$", alpha=0.5, **kwargs)

    comb = np.array(y_1 + y_0)

    plt.title(title)
    plt.xlabel("$\\mu(D)$")
    plt.xscale("log")
    plt.ylabel(desc)
    perc = np.percentile(comb, 99)
    mask=comb<perc
    if mask.any():
        plt.ylim(top = 1.2 * np.max(comb[mask]))
    plt.legend(loc="upper right")
    plt.savefig("plots/stat_" + filename + ".pdf")
    try:
        plt.yscale("log")
        plt.ylim(bottom=np.min(comb))
        if name == "FWHM":
            plt.ylabel("FWHM")
        plt.savefig("plots/stat_" + filename + "_logscale.pdf")
    except:
        pass
    plt.close()

    div = []
    for stat in stats:
        y_0_ = stat[0][name]
        y_1_ = stat[1][name]
        div.append(np.divide(y_1_, y_0_))

    ratio_string= desc + "$(D+\\gamma_{"+subscript+"})$ / " + desc + "$(D)$"
    plt.scatter(x, div, color="green", s=5, marker="o", alpha=0.5, label=ratio_string, **kwargs)
    plt.title(title)
    plt.xlabel("$\\mu(D)$")
    plt.xscale(scale)
    if mask.any():
        plt.ylim([0, 3])
    plt.axhline(y=1, color="lightblue")
    #plt.ylabel(ratio_string)

    plt.savefig("plots/stat_" + filename + "_div.pdf")
    plt.close()

    x_0 = [stat[0]["mean"] for stat in stats]
    x_1 = [stat[1]["mean"] for stat in stats]
    y_0 = [stat[0][name] for stat in stats]
    y_1 = [stat[1][name] for stat in stats]
    plt.scatter(x_0, y_0, color="cyan", s=5, marker="o", label="$D$", alpha=0.5)
    plt.scatter(x_1, y_1, color="tomato", s=5, marker="o", label="$D+\\gamma_{" + subscript + "}$", alpha=0.5)
    plt.xlabel("$\\mu$")
    plt.xscale(scale)
    plt.ylabel(desc)
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig("plots/stat_" + filename + "_abs" + ".pdf")
    plt.close()


    if name in ["std", "modal_deviation"]:
        #extremely ugly code to do minimaly less work
        plt.xscale(scale)
        plt.xlabel("$\\mu(D)$")
        plt.ylabel("seconds")
        df_ref = read_already_modified_from(["dataframes/dia_4k_1sTimeout_wlan_2__8.8.8.8_2022_05_11_19_51.csv"])
        plt.scatter(x, y_0, color="lightblue", s=5, marker="o", alpha=0.5, label="$\sigma(D)$",**kwargs)
        colors = ["yellow", "goldenrod", "orange", "red", "firebrick"]
        for hash in hashes + ["sha3_512_0"]:
            mean = np.mean(df_ref[0]["channel_noiseopenssl_" + hash])
            plt.axhline(y = mean, color=colors[0], label="$\\mu(\\gamma_{" + get_latex_from_hash_name(hash) + "})$")
            colors = colors[1:]
        plt.yscale("log")
        plt.ylim(top=0.5, bottom=1e-5)
        plt.legend(loc="upper left")

        plt.savefig("plots/stat_" + filename + "_diff.pdf")
        plt.close()

    if do_log:
        for stat in stats:
            x = stat[0]["mean"]
            y_0 = stat[0][name]
            y_1 = stat[1][name]
            plt.scatter(x, np.divide(y_1, y_0), color="green", s=10, marker="x", **kwargs)
        plt.title(name + " log")
        plt.xlabel("mean without cc")
        plt.ylim([0.1, 5])
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel(name + " with cc / " + name + " without cc")
        plt.close()

def get_stat_pairs(df, sec_column="modifiedopenssl_sha256_0"):
    return (create_statistical_data(df["ping time"]),
            create_statistical_data(df[sec_column]))

def plot_stats_for_files(files, prefix,
                         sec_column="modifiedopenssl_sha3_256_0",
                         scale="log", subscript=""):
    dfs = read_already_modified_from(files)

    hists_normal = []
    hists_modified = []

    stats = []

    stats = Parallel(n_jobs=7)(delayed(get_stat_pairs)(df, sec_column=sec_column) for df in dfs)

    title_map = {

    }

    desc_map = {

    }

    for c in stats[0][0].index:
        print(c)
        title = title_map.get(c, "")
        plot_stat(stats, c, prefix + c, title=title if title else "", desc=desc_map.get(c, ""), scale=scale, subscript=subscript)

    x_0 = [stat[0]["mode"] for stat in stats]
    x_1 = [stat[1]["mode"] for stat in stats]
    y_0 = [stat[0]["suessmann"] for stat in stats]
    y_1 = [stat[1]["suessmann"] for stat in stats]
    plt.scatter(x_0, y_0, color="cyan", s=5, marker="o", label="$D$", alpha=0.5)
    plt.scatter(x_1, y_1, color="tomato", s=5, marker="o", label="$D+\\gamma_{"+subscript+"}$", alpha=0.5)
    plt.title("Mode v. Süssmann measure")
    plt.xlabel("mode")
    plt.xscale(scale)
    plt.ylabel("$\\delta$")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig("plots/stat_" + prefix + "_suessman_modal_abs" + ".pdf")
    plt.close()

    x_0 = [stat[0]["mean"] for stat in stats]
    x_1 = [stat[1]["mean"] for stat in stats]
    y_0 = [stat[0]["mode"] for stat in stats]
    y_1 = [stat[1]["mode"] for stat in stats]
    plt.scatter(x_0, y_0, color="cyan", s=5, marker="o", label="$D$", alpha=0.5)
    plt.scatter(x_1, y_1, color="tomato", s=5, marker="o", label="$D+\\gamma_{"+subscript+"}$", alpha=0.5)
    plt.title("Mean v. Mode measure")
    plt.xlabel("$\\mu$")
    plt.xscale(scale)
    plt.ylabel("$mode$")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig("plots/stat_" + prefix + "_mean_mode" + ".pdf")
    plt.close()


for hash in hashes:
    plot_stats_for_files(glob.glob("dataframes/*192.168.1.1_*"), hash + "0LAN_", scale="linear", sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
    plot_stats_for_files(glob.glob("dataframes/*wlan__66.211.175.229*"), hash + "0long_", scale="linear", sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
    plot_stats_for_files(glob.glob("dataframes/*8.8.8.8*"), hash + "0google_", scale="linear", sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
    plot_stats_for_files(glob.glob("dataframes/*australia*"), hash + "0aus_", scale="linear", sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
    plot_stats_for_files(glob.glob("dataframes/*"), hash + "0all_", scale="log", sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
    plot_stats_for_files(glob.glob("RIPE/modified_data_1/*"), hash, scale="log", sec_column="openssl_" + hash, subscript=get_latex_from_hash_name(hash))

hash = "sha3_512_0"

plot_stats_for_files(glob.glob("dataframes/*192.168.1.1_*"), hash + "0LAN_", scale="linear",
                     sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
plot_stats_for_files(glob.glob("dataframes/*wlan__66.211.175.229*"), hash + "0long_", scale="linear",
                     sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
plot_stats_for_files(glob.glob("dataframes/*8.8.8.8*"), hash + "0google_", scale="linear",
                     sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
plot_stats_for_files(glob.glob("dataframes/*australia*"), hash + "0aus_", scale="linear",
                     sec_column="modifiedopenssl_" + hash, subscript=get_latex_from_hash_name(hash))
plot_stats_for_files(glob.glob("dataframes/*"), hash + "0all_", scale="log", sec_column="modifiedopenssl_" + hash,
                     subscript=get_latex_from_hash_name(hash))

hash = "sha512_0"

plot_stats_for_files(glob.glob("RIPE/modified_data_1/*"), hash, sec_column="openssl_" + hash,
                     subscript=get_latex_from_hash_name(hash))


