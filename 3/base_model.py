import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras import optimizers
import re
import json

from cache import Cache


class TestCallback(Callback):
    def __init__(self, model, test_data):
        self._model = model
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, mae, acc = self.model.evaluate(x, y, verbose=0)
        self._model.acc_history.append(acc)


class BaseModel:
    categories = ['guest', 'nonguest']
    n_out = 1

    def __init__(self, params):
        self.params = params
        self.cache = Cache(type(self).__name__, self.version)


    def get_y(self, d):
        return np.where(d['moderated_role'] == 'guest', 'guest', 'nonguest')

    def encode_y(self, y):
        y = np.where(y == 'guest', [1], [0])
        return y
        #dummies = pd.get_dummies(y)
        #return dummies.as_matrix()

    def decode_y(self, y):
        y = y.transpose()[0]
        return np.where(y > 0.5, "guest", "nonguest")


    def reset(self):
        self.model = self.create_model()

    def compile(self):
        Optimizer = getattr(optimizers, self.params['opt'])
        opt = Optimizer(lr = self.params['lr'])
        return self.model.compile(optimizer=opt,
                                  loss='binary_crossentropy',
                                  metrics=['mae', 'acc'])

    def fit(self, x, y, test_x, test_y):
        self.acc_history = []
        validation = (test_x, self.encode_y(test_y))
        return self.model.fit(x, self.encode_y(y),
                              epochs = self.params['epochs'], batch_size = self.params.get('bs', 256),
                              validation_data = validation,
                              callbacks=[TestCallback(self, validation)])

    def predict(self, x):
        return self.decode_y(self.model.predict(x))

    def predict_undecoded(self, x):
        return self.model.predict(x)

    def debug(self, x):
        p = self.model.predict(x)
        import code; code.interact(local=dict(globals(), **locals()))


    def params_str(self):
        def sanitize(s):
            s = str(s)
            s = re.sub("^[^\w.]+", "", s)
            s = re.sub("[^\w.]+$", "", s)
            s = re.sub("[^\w.]+", "-", s)
            return s

        items = self.params.items()
        items = sorted(items, key = lambda i: i[0])
        parts = map(lambda i: "{}={}".format(i[0], sanitize(i[1])), items)
        return "_".join(parts)

    def params_json(self):
        return json.dumps(self.params)
