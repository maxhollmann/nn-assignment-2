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
    # filter out data we won't use
    ind = d['moderated_role'] == 'guest'
    ind = np.logical_or(ind, d['moderated_role'] == 'host')
    ind = np.logical_or(ind, d['moderated_role'] == 'neither')
    d = d[ind]


    pd.options.mode.chained_assignment = None
    # booleans
    d['name_in_title'                    ] = d['name_in_title'                    ].map({'false': 0, 'true': 1})
    d['name_in_description'              ] = d['name_in_description'              ].map({'false': 0, 'true': 1})
    d['name_in_podcast_author'           ] = d['name_in_podcast_author'           ].map({'false': 0, 'true': 1})
    d['name_in_podcast_managing_editor'  ] = d['name_in_podcast_managing_editor'  ].map({'false': 0, 'true': 1})
    d['with_before_name_in_title'        ] = d['with_before_name_in_title'        ].map({'false': 0, 'true': 1})
    d['name_is_first_word_in_description'] = d['name_is_first_word_in_description'].map({'false': 0, 'true': 1})

    # numbers
    d['times_mentioned'                    ] = d['times_mentioned'                    ].astype(np.float)
    d['percentage_of_episodes_mentioned_on'] = d['percentage_of_episodes_mentioned_on'].astype(np.float)
    d['number_of_people_mentioned'] = d['number_of_people_mentioned'].astype(np.float)
    #d['position_of_name_in_title'] = d['position_of_name_in_title'].astype(np.float)
    #d['position_of_name_in_description'] = d['position_of_name_in_description'].astype(np.float)
    d['percentage_of_episodes_of_podcast_mentioned_on'] = d['percentage_of_episodes_of_podcast_mentioned_on'].astype(np.float)

    pd.options.mode.chained_assignment = 'warn'

    for k in d:
        print("{: <50}: {: <40} ({})".format(k, d[k][0], type(d[k][0])))

    return d


categories = ['guest', 'nonguest']

nlp = spacy.load('en')
