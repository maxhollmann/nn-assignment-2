import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import os

from cache import Cache
from plotting import plot_confusion_matrix

# model consists of
# - preprocess_data   can filter out data points that aren't processable by the model
# - get_x             generate the X matrix
# - get_y             generate the Y matrix
# - encode_y          encode Y to model outputs
# - decode_y          decode model outputs to Y
# - model             the keras model
#   - fit
#   - predict
# - version           used to invalidate cache

class Test:
    def __init__(self, name, module, data, use_cache=True):
        self.model_name    = name
        self.module        = module
        self.orig_data     = data
        self.data          = module.preprocess_data(data)
        self.model         = module.model
        self.model_version = module.version
        self.cache         = Cache(self.model_name, self.model_version, use = use_cache)



    def run(self):
        print("Testing '{}' version {}".format(self.model_name, self.model_version))
        y = self.module.get_y(self.data)
        x = self.cache.fetch("x", lambda: self.module.get_x(self.data))

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['mae', 'acc'])

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        self.model.fit(x_train, self.module.encode_y(y_train), epochs = 15)
        pred_test = self.module.decode_y(self.model.predict(x_test))
        pred_train = self.module.decode_y(self.model.predict(x_train))
        print("{} / {}".format(np.sum(y_train == "guest"), len(y_train)))

        self.accuracy_test = np.sum(pred_test == y_test) / len(y_test)
        self.accuracy_train = np.sum(pred_train == y_train) / len(y_train)


        self.cnf_matrix = metrics.confusion_matrix(y_test, pred_test, labels = ["guest", "nonguest"])

    def store_plots(self):
        fig = plt.figure()
        plot_confusion_matrix(self.cnf_matrix,
                              classes = ["guest", "nonguest"],
                              title='Confusion matrix for {}'.format(self.model_name),
                              normalize = True)

        plt.savefig(self.path("confusion_matrix.png"))
        plt.close()

    def path(self, filename):
        dir = os.path.join("out", self.model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        return os.path.join(dir, filename)
