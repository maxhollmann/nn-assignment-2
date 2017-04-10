from keras.models import Sequential
from keras.layers import Dense, Activation

from preprocessing import *

import re


version = 1

n_words = 5

def get_x_row(row):
    name_re = re.compile(re.escape(row['name']), re.IGNORECASE)
    desc = name_re.sub("THENAME", row['description'])
    print(row['name'])
    doc = nlp(desc)
    doclist = [w.string.strip() for w in doc]
    pad_vector = nlp('.').vector
    try:
        ind = doclist.index("THENAME")
    except ValueError:
        ind = -100

    out = []
    for i in list(range(ind - n_words, ind)) + list(range(ind + 1, ind + n_words + 1)):
        if i < 0 or i >= len(doc):
            out.append(pad_vector)
        else:
            out.append(doc[i].vector)

    return np.hstack(out)


def get_x(d):
    return np.row_stack([get_x_row(row) for index, row in d.iterrows()])

def create_model():
    return Sequential([
        Dense(100, input_dim=2*n_words*300),
        Activation('relu'),
        Dense(100),
        Activation('relu'),
        Dense(len(categories)),
        Activation('softmax'),
    ])
