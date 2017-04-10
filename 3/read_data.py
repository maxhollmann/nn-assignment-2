import csv
import pandas as pd


def read_data(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        d = pd.DataFrame(list(reader))



    return d
