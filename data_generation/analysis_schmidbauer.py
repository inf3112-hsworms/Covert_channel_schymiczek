import pandas

from data_generation.ping_statistics import plot

plot(pandas.read_csv("../diagnostics/dia__9.9.9.9_2022_04_26_19_12.csv")["ping time"])