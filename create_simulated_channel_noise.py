import glob
import hashlib

from ml_tools.ml_util import get_and_modify_data_from

#measure_all("ip_list", pings=4000, return_type="file", name_prefix="dia_4k_1sTimeout_wlan_2_", savedir="./diagnostics_2")

get_and_modify_data_from(glob.glob("diagnostics_2/*195.12.50.155*"),
                         [hashlib.md5, hashlib.sha256, hashlib.sha3_256,
                          hashlib.sha384, hashlib.sha3_512], savemode=True, save_to="dataframes")
