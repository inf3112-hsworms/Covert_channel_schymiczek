import glob

import numpy as np
import pandas

from graphs_comparison_channel_no_channel import compare_min_max
from ml_tools.ml_util import read_already_modified_from



LAN = read_already_modified_from(glob.glob("dataframes_2/*_lan_*"))

all_LAN = pandas.concat(LAN)

if __name__ == "__main__":
    c_2 = ["openssl_md5_cc_0", "openssl_sha256_cc_3", "openssl_sha3_256_cc_0", "openssl_sha384_cc_0", "openssl_sha3_512_cc_0"]
    labels = ["$\gamma_{md5}$", "$\gamma_{sha256}$", "$\gamma_{sha3-256}$", "$\gamma_{sha384}$", "$\gamma_{sha3-512}$"]

    for i in range(len(c_2)):
        c = c_2[i]
        compare_min_max([all_LAN], do_compare=False, filename="_CC_compare_complete" + c, columns=[c],
                    colors_light=["red"],
                    labels=[labels[i]],
                    hist_func=lambda x: np.histogram(x, bins=1000, range=[0, 0.003]))
