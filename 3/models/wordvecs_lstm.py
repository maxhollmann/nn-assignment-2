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

    params = []
    params.append({'opt': 'SGD', 'epochs': 20, 'lr': 0.03, 'n_words': 10})


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
                print("Name not found!")
                ind = -100

            out = []
            for i in list(range(ind - self.params['n_words'], ind)) + list(range(ind + 1, ind + self.params['n_words'] + 1)):
                if i < 0 or i >= len(doc):
                    out.append(pad_vector)
                else:
                    out.append(doc[i].vector)

            return np.array(out)


        def get():
            import code; code.interact(local=dict(globals(), **locals()))
            return np.array([get_row(row) for index, row in d.iterrows()])

        k = "x/n_words={}".format(self.params['n_words'])
        return self.cache.fetch(k, get)

    def create_model(self):
        model = Sequential()
        #model.add(Dense(input_dim=2*self.params['n_words']*300))

        model.add(LSTM(50, input_dim=(300)))

        model.add(Dense(self.n_out, activation='sigmoid'))

        return model
