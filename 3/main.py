import argparse
import pkgutil

import models
from read_data import read_data
from testing import Test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", default="data.csv", help="CSV file")
    parser.add_argument("--only", dest="only", nargs="+", help="Only run specified models")
    parser.add_argument("--exclude", dest="exclude", nargs="+", help="Don't run specified models")
    parser.add_argument('--no-cache', dest='cache', action='store_false')
    parser.set_defaults(cache=True)
    args = parser.parse_args()

    data = read_data(args.data)

    for _, modname, _ in pkgutil.iter_modules(models.__path__):
        if args.only and modname not in args.only:
          continue
        if args.exclude and modname in args.exclude:
          continue

        module = __import__("models." + modname, fromlist = "dummy")
        test = Test(modname, module, data, use_cache = args.cache)
        test.run()
        print("Accuracy: {:.4f} (test) / {:.4f} (train)".format(test.accuracy_test, test.accuracy_train))
        test.store_plots()



if __name__ == "__main__":
    main()
