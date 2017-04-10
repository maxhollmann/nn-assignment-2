import numpy as np
import pandas as pd
import spacy

def get_y(d):
    return np.where(d['moderated_role'] == 'guest', 'guest', 'nonguest')

def encode_y(y):
    y = np.where(y == 'guest', 'guest', 'nonguest')
    dummies = pd.get_dummies(y)
    return dummies.as_matrix()

def decode_y(y):
    return np.where(np.argmax(y, 1) == 1, "nonguest", "guest")

def preprocess_data(d):
    ind = d['moderated_role'] == 'guest'
    ind = np.logical_or(ind, d['moderated_role'] == 'host')
    ind = np.logical_or(ind, d['moderated_role'] == 'neither')
    return d[ind]

categories = ['guest', 'nonguest']

nlp = spacy.load('en')
