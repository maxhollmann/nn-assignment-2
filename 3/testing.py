import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import paths
from glob import glob

from plotting import plot_confusion_matrix
from logger import CsvLogger


# model consists of
# - get_x             generate the X matrix
# - get_y             generate the Y matrix
# - encode_y          encode Y to model outputs
# - decode_y          decode model outputs to Y
# - model             the keras model
#   - fit
#   - predict
# - version           used to invalidate cache

class TestCase:
    def __init__(self, test, params):
        self.model_name = test.model_name
        self.model = test.Model(params)
        self.data = test.data
        self.splits = test.splits

        self.accuracy_test  = None
        self.accuracy_train = None
        self.cnf_matrix     = None

    def run(self):
        y = self.model.get_y(self.data)
        x = self.model.get_x(self.data)

        accuracy_test  = []
        accuracy_train = []
        cnf_matrix     = []
        for i, (train, test) in enumerate(self.splits):
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]

            self.model.reset()
            self.model.compile()
            self.model.fit(x_train, y_train, x_test, y_test)

            accuracy_test.append(self.model.acc_history)

            pred_test  = self.model.predict(x_test)
            cnf_matrix.append(metrics.confusion_matrix(y_test, pred_test, labels = self.model.categories))
            self.store_output(i, test)
            self.store_model(i)

        self.accuracy_test  = np.mean(accuracy_test, axis = 0)
        self.accuracy_train = 0
        self.cnf_matrix     = np.mean(cnf_matrix, axis=0)

    def log_acc(self):
        csv = CsvLogger(self.path("accuracy.csv"),
                        ["model", "params", "epoch", "acc"])
        for i, acc in enumerate(self.model.acc_history):
            csv.log([self.model_name, self.model.params_json(), i, acc])

    def store_model(self, i):
        self.model.model.save(self.path("model_{}.hdf5".format(i)))

    def store_output(self, isplit, test):
        x = self.model.get_x(self.data)
        x_test = x[test]
        out = self.model.predict_undecoded(x_test)

        csv = CsvLogger(self.path("output_{}.csv".format(isplit)),
                        ["mention_id", "nn_out"])
        for i, o in enumerate(out):
            csv.log([self.data.iloc[test[i], :]['mention_id'], o[0]])


    def store_plots(self):
        fig = plt.figure()
        plot_confusion_matrix(self.cnf_matrix,
                              classes = self.model.categories,
                              title='Confusion matrix for {}'.format(self.model_name),
                              normalize = True)

        plt.savefig(self.path("confusion_matrix.png"))
        plt.close()

    def path(self, filename):
        return paths.f(self.model_name, str(self.model.version),
                       f = "{}__{}".format(self.model.params_str(), filename))


class Test:
    def __init__(self, name, module, data, splits):
        self.model_name    = name
        self.Model         = module.Model
        self.data          = data
        self.splits        = splits


    def run_all(self):
        print("Testing '{}' version {} - {} configurations".format(
            self.model_name, self.Model.version, len(self.Model.params)))

        self.cases = []
        for params in self.Model.params:
            case = TestCase(self, params)
            self.cases.append(case)

        for i, case in enumerate(self.cases):
            print("    {}/{}   params: {}".format(i + 1, len(self.cases), str(case.model.params)))
            try:
                if True or len(glob(case.path("*"))) == 0:
                    case.run()
                    case.log_acc()
                    case.store_plots()
                else:
                    print("Found existing output for model, skipping")
            except Exception as e:
                import traceback
                print("\n")
                traceback.print_exc()
                print("\n")
