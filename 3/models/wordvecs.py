import numpy as np
import spacy
import re
from keras.models import Sequential
from keras.layers import Dense, Activation

from base_model import BaseModel

nlp = spacy.load('en')


class Model(BaseModel):
    version = 1

    params = [
        {'n_words': 5},
        {'n_words': 10},
    ]

    def get_x_row(self, row):
        name_re = re.compile(re.escape(row['name']), re.IGNORECASE)
        desc = name_re.sub("THENAME", row['description'])
        doc = nlp(desc)
        doclist = [w.string.strip() for w in doc]
        pad_vector = nlp('.').vector
        try:
            ind = doclist.index("THENAME")
        except ValueError:
            ind = -100

        out = []
        for i in list(range(ind - self.params['n_words'], ind)) + list(range(ind + 1, ind + self.params['n_words'] + 1)):
            if i < 0 or i >= len(doc):
                out.append(pad_vector)
            else:
                out.append(doc[i].vector)

        return np.hstack(out)


    def get_x(self, d):
        def get():
            return np.row_stack([self.get_x_row(row) for index, row in d.iterrows()])

        k = "x/n_words_{}".format(self.params['n_words'])
        return self.cache.fetch(k, get)

    def create_model(self):
        return Sequential([
            Dense(1000, input_dim=2*self.params['n_words']*300),
            Activation('relu'),
            Dense(len(self.categories)),
            Activation('softmax'),
        ])
