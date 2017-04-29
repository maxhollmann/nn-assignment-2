import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os

from plotting import plot_confusion_matrix
from logger import log


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
    def __init__(self, model, data, splits):
        self.model = model
        self.data = data
        self.splits = splits

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
            self.model.fit(x_train, y_train)

            pred_test  = self.model.predict(x_test)
            pred_train = self.model.predict(x_train)

            accuracy_test. append(np.sum(pred_test == y_test) / len(y_test))
            accuracy_train.append(np.sum(pred_train == y_train) / len(y_train))
            cnf_matrix.append(metrics.confusion_matrix(y_test, pred_test, labels = self.model.categories))

        self.accuracy_test  = np.mean(accuracy_test)
        self.accuracy_train = np.mean(accuracy_train)
        self.cnf_matrix     = np.mean(cnf_matrix, axis=0)


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
            case = TestCase(self.Model(params), self.data, self.splits)
            self.cases.append(case)

        for case in self.cases:
            print("    params: {}".format(case.model.params_str()))
            try:
                case.run()
                log("{} - {}: {} / {}".format(
                    self.model_name, case.model.params_str(),
                    case.accuracy_test, case.accuracy_train))
            except Exception as e:
                import traceback
                print("\n")
                traceback.print_exc()
                print("\n")


    def store_plots(self):
        fig = plt.figure()
        plot_confusion_matrix(self.cnf_matrix,
                              classes = self.Model.categories,
                              title='Confusion matrix for {}'.format(self.model_name),
                              normalize = True)

        plt.savefig(self.path("confusion_matrix.png"))
        plt.close()

    def path(self, filename):
        dir = os.path.join("out", self.model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return os.path.join(dir, filename)
