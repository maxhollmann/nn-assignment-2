import argparse
import pkgutil

from sklearn.model_selection import StratifiedKFold

import models
from read_data import read_data
from testing import Test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", default="data.csv", help="CSV file")
    parser.add_argument("--only", dest="only", nargs="+", help="Only run specified models")
    parser.add_argument("--exclude", dest="exclude", nargs="+", help="Don't run specified models")
    args = parser.parse_args()

    data = read_data(args.data)


    splits = StratifiedKFold(n_splits=5, shuffle=True)
    splits = [s for s in splits.split(data['title'], data['moderated_role'])]

    tests = []
    for _, modname, _ in pkgutil.iter_modules(models.__path__):
        if args.only and modname not in args.only:
          continue
        if args.exclude and modname in args.exclude:
          continue

        module = __import__("models." + modname, fromlist = "dummy")
        test = Test(modname, module, data, splits)
        tests.append(test)

    for test in tests:
        test.run_all()

    print("\n")
    for test in tests:
        for case in test.cases:
            print("{: <16} with {: <50} Accuracy: {:.4f}".format(
                test.model_name, case.model.params_str(), case.accuracy_test[-1]
            ))



if __name__ == "__main__":
    main()
