from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np

from base_model import BaseModel



class Model(BaseModel):
    version = 1

    params = [
        {'opt': optimizers.SGD, 'epochs': 20, 'lr': 0.001},
    ]


    def create_model(self):
        return Sequential([
            Dense(self.n_out, input_dim=7),
            Activation('softmax'),
        ])

    def get_x(self, d):
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
