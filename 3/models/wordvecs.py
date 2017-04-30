import numpy as np
import spacy
import re
from keras.models import Sequential
from keras.layers import Dense, Activation

from base_model import BaseModel

nlp = spacy.load('en')


class Model(BaseModel):
    version = 1

    #params = [
    #    {'opt': optimizers.SGD, 'epochs': 10, 'lr': 0.1, 'n_words': 3, 'hidden': [100]},
    #    {'opt': optimizers.SGD, 'epochs': 100, 'lr': 0.1, 'n_words': 3, 'hidden': [100]},
    #    {'opt': optimizers.SGD, 'epochs': 100, 'lr': 0.05, 'n_words': 3, 'hidden': [100]},
    #    {'opt': optimizers.SGD, 'epochs': 500, 'lr': 0.05, 'n_words': 3, 'hidden': [100]},
    #]

    # params = []
    # for epochs in [10, 100, 250]:
    #     for lr in [0.01, 0.05, 0.1]:
    #         for n_words in [2, 3, 5]:
    #            for l1 in [[10], [100], [1000]]:
    #                for l2 in [[], l1]:
    #                    params.append({'opt': 'SGD', 'epochs': epochs, 'lr': lr, 'n_words': n_words, 'hidden': l1 + l2})

    params = []
    for epochs in [250, 300, 400]:
        for lr in [0.025, 0.05, 0.075]:
            for n_words in [5, 10]:
               for l1 in [[1000], [1500], [2000]]:
                   for nhidden in [1, 2, 3]:
                       params.append({'bs': 256, 'opt': 'SGD', 'epochs': epochs, 'lr': lr, 'n_words': n_words, 'hidden': l1 * nhidden})


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
            Dense(self.n_out),
            Activation('sigmoid'),
        ])

        return Sequential(layers)
