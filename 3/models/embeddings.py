import numpy as np
import spacy
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence, text
from collections import Counter

from base_model import BaseModel

nlp = spacy.load('en')


class Model(BaseModel):
    version = 1

    params = [
        # {'opt': 'SGD', 'lr': 0.05, 'epochs': 1, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'hidden': [1000]},
        # {'opt': 'SGD', 'lr': 0.05, 'epochs': 2, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'hidden': [1000]},
        # {'opt': 'SGD', 'lr': 0.05, 'epochs': 3, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'hidden': [1000]},

        # {'opt': 'SGD', 'lr': 0.2, 'epochs': 1, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'hidden': [1000]},
        # {'opt': 'SGD', 'lr': 0.9, 'epochs': 1, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'hidden': [1000]},

        # {'opt': 'SGD', 'lr': 0.1, 'epochs': 2, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'dropout': 0.2},

        # {'opt': 'SGD', 'lr': 0.1, 'epochs': 2, 'bt': 32, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 200, 'dropout': 0.2},

        {'opt': 'SGD', 'lr': 0.1, 'epochs': 4, 'bt': 32, 'embedlen': 32, 'vocabsize': 4000, 'textlen': 500, 'dropout': 0.2},
    ]


    def get_x(self, d):
        def row_tokens(row):
            def tokenize(s):
                return np.array([w.lower_ for w in nlp(s)])

            name_parts = row['name'].lower().split()

            text = "{}\n\n{}".format(row['title'], row['description']).strip()
            tokens = tokenize(text)
            for n in name_parts:
                tokens = np.where(tokens == n, "thename", tokens)

            return tokens


        def get_vocab(tokenized, n):
            c = Counter()
            for t in tokenized:
                c.update(t)
            mc = c.most_common(n - 1)
            vocab = {w: i + 1 for i, (w, f) in enumerate(mc)}
            return np.vectorize(lambda word: vocab.get(word, 0))

        def get_indexed():
            tokenized = [row_tokens(row) for index, row in d.iterrows()]
            vocab = get_vocab(tokenized, self.params['vocabsize'])
            indexed = np.array([vocab(row) for row in tokenized])
            indexed = sequence.pad_sequences(indexed, self.params['textlen'])
            return indexed

        k = "x_indexed/vocabsize={}_textlen={}".format(self.params['vocabsize'], self.params['textlen'])
        indexed = self.cache.fetch(k, get_indexed)
        return indexed


    def create_model(self):
        embedding = Embedding(self.params['vocabsize'],
                              self.params['embedlen'],
                              #mask_zero = True,
                              input_length = self.params['textlen'])



        model = Sequential()
        model.add(embedding)
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(self.n_out, activation='sigmoid'))

        return model
