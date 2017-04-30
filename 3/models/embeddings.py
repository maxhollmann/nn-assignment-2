import numpy as np
import spacy
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text

from base_model import BaseModel

nlp = spacy.load('en')


class Model(BaseModel):
    version = 1

    params = [{}]


    def get_x(self, d):
        def get_row(row):
            def tokenize(s):
                return np.array([w.lower_ for w in nlp(s)])

            name_parts = row['name'].lower().split()

            text = "{}\n\n{}".format(row['title'], row['description']).strip()
            tokens = tokenize(text)
            for n in name_parts:
                tokens = np.where(tokens == n, "thename", tokens)
            return tokens


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
            r = d.loc[0, :]
            r = get_row(r)
            return np.row_stack([get_row(row) for index, row in d.iterrows()])

        import code; code.interact(local=dict(globals(), **locals()))
        #
        #k = "x/n_words={}".format(self.params['n_words'])
        #return self.cache.fetch(k, get)

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
            Activation('relu'), Dense(self.n_out),
            Activation('softmax')
        ])

        return Sequential(layers)
