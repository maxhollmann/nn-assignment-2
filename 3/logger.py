import csv
import paths

def log(s, level = 'debug'):
    with open(paths.f("results.txt"), "a") as f:
        f.write(str(s) + "\n")
    print(s)

def csv_log(name, row, level = 'debug'):
    with open(name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(s)

class CsvLogger:
    def __init__(self, fname, header):
        self.fname = fname
        self.log(header, "w")

    def log(self, row, mode = "a"):
        with open(self.fname, mode) as f:
            writer = csv.writer(f)
            writer.writerow(row)
