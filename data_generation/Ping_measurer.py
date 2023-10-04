import datetime
import glob
import hashlib
import os

import pandas
from datetime import datetime
from scapy.all import sr, IP
from scapy.layers.inet import traceroute, ICMP

from ml_tools.ml_util import get_and_modify_data_from


def measure(ip, traceroute_attempts=32, pings=int(1e3),
            savedir="./diagnostics", name_prefix="dia_lan", return_type="file"):
    print(ip)
    traceroute_counter = {}
    for i in range(traceroute_attempts):
        res, unans = traceroute(ip, verbose=False)
        trace=res.get_trace()
        for addr in trace:
            traceroute_counter[addr] = traceroute_counter.get(addr, 0) + len(trace[addr])
    avg_traceroute={}
    for cnt in traceroute_counter:
        avg_traceroute[cnt] = traceroute_counter[cnt] / traceroute_attempts

    print("avg traceroute: " + str(avg_traceroute))

    array = []

    for i in range(pings):
        print(i)
        ans, unans = sr(IP(dst=ip) / ICMP(), verbose=False, timeout=1)
        for s, r in ans:
            if IP in r and r[IP].src == ip:
                time = r.time - s.sent_time
                print(time)
                to_be_added = [s.sprintf("%IP.dst%"), avg_traceroute[s.sprintf("%IP.dst%")], time]
                array.append(to_be_added)

    df = pandas.DataFrame(array, columns=["ip", "avg traceroute", "ping time"])
    print(df)
    if array is not None:
        if return_type=="file":
            filename=f"{savedir}/{name_prefix}_{ip}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
        elif return_type=="df":
            return df

def measure_all(ip_file, **kw_args):
    ips = open(ip_file).readlines()

    for ip in ips:
        measure(ip.replace(" ", "").replace("\n", ""), **kw_args)


if __name__ == "__main__":
    measure_all("../ip_list", pings=4000, return_type="file", name_prefix="dia_test_lag_lan_normal", savedir="./diagnostics_2")
    get_and_modify_data_from(glob.glob("diagnostics_2/*dia_test_lag_lan_normal*"),
                             [hashlib.md5, hashlib.sha256, hashlib.sha3_256,
                              hashlib.sha384, hashlib.sha3_512], savemode=True, save_to="dataframes")

