import glob
import json
import os.path

import numpy


def extract_from_RIPE(file):
    with open(file, "r") as file_:
        for line in file_:
            obj = json.loads(line)
            src = obj.get("from", False)
            dst = obj.get("dst_name", False)
            if src and dst:
                ttl = obj.get("ttl", numpy.NAN)
                size = obj.get("size", numpy.NAN)
                results = obj.get("result")

                if len(results) >= 10 and results[0].get("rtt", False):
                    print(line)
                    write_to = "RIPE/separated_data/" + src + "->" + dst
                    if not os.path.exists(write_to):
                        first_write = open(write_to, "w")
                        first_write.write("src, dst, ttl, size, ping time \n")
                        first_write.close()

                    write_to = open(write_to, "a")

                    for res in results:
                        if res.get("rtt", False):
                            write_to.write(f"{src}, {dst}, {ttl}, {size}, {res['rtt']}\n")

                    write_to.close()


for file in glob.glob("RIPE/ping*"):
    print("----------------------------------------------------")
    print(file)
    print("----------------------------------------------------")
    extract_from_RIPE(file)
