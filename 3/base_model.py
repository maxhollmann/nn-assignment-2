import numpy as np
import pandas as pd
import re

from cache import Cache


class BaseModel:
    categories = ['guest', 'nonguest']

    def __init__(self, params):
        self.params = params
        self.cache = Cache(type(self).__name__, self.version)


    def get_y(self, d):
        return np.where(d['moderated_role'] == 'guest', 'guest', 'nonguest')

    def encode_y(self, y):
        y = np.where(y == 'guest', 'guest', 'nonguest')
        dummies = pd.get_dummies(y)
        return dummies.as_matrix()

    def decode_y(self, y):
        return np.where(np.argmax(y, 1) == 1, "nonguest", "guest")


    def reset(self):
        self.model = self.create_model()

    def compile(self):
        return self.model.compile(optimizer='rmsprop',
                                  loss='categorical_crossentropy',
                                  metrics=['mae', 'acc'])

    def fit(self, x, y):
        return self.model.fit(x, self.encode_y(y), epochs = 15)

    def predict(self, x):
        return self.decode_y(self.model.predict(x))


    def params_str(self):
        return str(self.params)

    def params_filename_part(self):
        def sanitize(s):
            s = str(s)
            s = re.sub("^\\W+", "", s)
            s = re.sub("\\W+$", "", s)
            s = re.sub("\\W+", "-", s)
            return s

        parts = map(lambda i: "{}={}".format(i[0], sanitize(i[1])), self.params.items())
        return "_".join(parts)
