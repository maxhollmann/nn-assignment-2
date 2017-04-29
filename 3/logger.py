import csv

def log(s, level = 'debug'):
    with open("out/results.txt", "a") as f:
        f.write(str(s) + "\n")
    print(s)

def csv_log(filename, row, level = 'debug'):
    with open("out/{}.csv".format(filename), "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(s)

class CsvLogger:
    def __init__(self, name, header):
        self.fname = "out/{}.csv".format(name)
        self.log(header, "w")

    def log(self, row, mode = "a"):
        with open(self.fname, mode) as f:
            writer = csv.writer(f)
            writer.writerow(row)
