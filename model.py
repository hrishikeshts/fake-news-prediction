import numpy as np
import re
from numpy import asarray

import tensorflow_hub as hub

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')


def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)


import pandas as pd

df_data = pd.read_csv('train_stances.csv')

df_body = pd.read_csv('train_bodies.csv')

final_df = df_data.merge(df_body, on='Body ID', how='inner')

final_df['Stance'] = final_df['Stance'].replace('disagree', 'irrelevant')

final_df = final_df.iloc[np.random.permutation(len(final_df))]

f1 = final_df[final_df['Stance'] == 'agree'][:700]
f2 = final_df[final_df['Stance'] == 'discuss'][:450]
f3 = final_df[final_df['Stance'] == 'unrelated'][:700]
f4 = final_df[final_df['Stance'] == 'irrelavent'][:700]

len(f4)

final_df = pd.concat([f1, f2, f3, f4])

final_df

del final_df['Body ID']

import pandas as pd
from numpy import asarray
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

final_df['Stance'] = final_df.Stance.astype('category').cat.codes


def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)


def cosines(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)


def test_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]

    return cosines(vec1, vec2)


max_size = 50

final_df['articleBody'] = final_df['articleBody'].str.split(n=max_size).str[:max_size].str.join(' ')

final_df['articleBody'] = final_df['articleBody'].apply(lambda x: [x])

final_df['Headline'] = final_df['Headline'].apply(lambda x: [x])

test_similarity(['David Haines Beheaded By ISIS, Execution Video Contains Threats Against UK: Reports'],
                ['Multiple unconfirmed sources report that Haines has been beheaded by ISIS.'])

k = []
for index, row in final_df.iterrows():
    k.append(test_similarity(row['Headline'], row['articleBody']))

st = list(final_df['Stance'])
dict_fk = {"Cosine_similarity": k, 'Stance': st}

df_preprocessed = pd.DataFrame(dict_fk)

x_train, x_test, y_train, y_test = train_test_split(df_preprocessed['Cosine_similarity'], df_preprocessed['Stance'],
                                                    shuffle=True)

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM


def get_keras_model():
    """Define the model."""
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model


model = get_keras_model()

# y_train = asarray(y_train, dtype="float32")
# y_test = asarray(y_test, dtype="float32")

model.fit(x_train, y_train, epochs=10)

model.save("fnd.h5")