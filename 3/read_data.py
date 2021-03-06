import csv
import numpy as np
import pandas as pd


def read_data(filename):
    with open(filename, encoding="utf8") as f:
        reader = csv.DictReader(f)
        d = pd.DataFrame(list(reader))

    return process(d)


def process(d):
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

    return d
