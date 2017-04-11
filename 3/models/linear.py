from keras.models import Sequential
from keras.layers import Dense, Activation

from preprocessing import *

version = 1

def get_x(d):
    x = np.column_stack([
        d['name_in_title'],
        d['name_in_description'],
        d['name_in_podcast_author'],
        d['with_before_name_in_title'],
        d['name_is_first_word_in_description'],
        d['times_mentioned'],
        d['percentage_of_episodes_mentioned_on'],
    ])
    return x


def create_model():
    return Sequential([
        Dense(7, input_dim=7),
        Activation('relu'),
        Dense(len(categories)),
        Activation('softmax'),
    ])
