from keras.models import Sequential
from keras.layers import Dense, Activation

from preprocessing import *

version = 1

def get_x(d):
    x = np.column_stack([
        np.where(d['name_in_title'] == 'true', 1, 0),
        np.where(d['name_in_description'] == 'true', 1, 0),
        np.where(d['name_in_podcast_author'] == 'true', 1, 0),
        np.where(d['with_before_name_in_title'] == 'true', 1, 0),
        np.where(d['name_is_first_word_in_description'] == 'true', 1, 0),
        d['times_mentioned'].astype(np.float),
        d['percentage_of_episodes_mentioned_on'].astype(np.float),
    ])
    return x



model = Sequential([
    Dense(7, input_dim=7),
    Activation('relu'),
    Dense(n_categories),
    Activation('softmax'),
])
