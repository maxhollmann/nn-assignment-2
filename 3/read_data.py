import csv
import pandas as pd


def read_data(filename):
    with open(filename, encoding="utf8") as f:
        reader = csv.DictReader(f)
        d = pd.DataFrame(list(reader))



    return d
