import numpy as np
import spacy
import re
from keras.models import Sequential
from keras.layers import Dense, Activation

from base_model import BaseModel

nlp = spacy.load('en')


class Model(BaseModel):
    version = 1

    params = []
    for n_words in [1, 3, 5, 10, 15]:
       for l1 in [[1], [10], [100], [1000]]:
           for l2 in [[], [1], [10], [100], [1000]]:
               params.append({'n_words': n_words, 'hidden': l1 + l2})



    def get_x(self, d):
        def get_row(row):
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


        def get():
            return np.row_stack([get_row(row) for index, row in d.iterrows()])

        k = "x/n_words={}".format(self.params['n_words'])
        return self.cache.fetch(k, get)

    def create_model(self):
        hidden = self.params['hidden']

        layers = [
            Dense(hidden[0], input_dim=2*self.params['n_words']*300)
        ]
        for h in hidden[1:]:
            layers.extend([
                Activation('relu'),
                Dense(h)
            ])
        layers.extend([
            Activation('relu'), Dense(len(self.categories)),
            Activation('softmax')
        ])

        return Sequential(layers)
