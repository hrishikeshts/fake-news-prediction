import os
from flask import Flask
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
app = Flask(__name__, static_folder='./build', static_url_path='/')

model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"fnd.h5"))
import numpy as np
import re
from numpy import asarray

import tensorflow_hub as hub

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')


def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)


import pandas as pd


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
import json
import requests
url = 'https://newsapi.org/v2/top-headlines?language=en&apiKey=f1fc1540925b40bd9bdc1aead5e83072&pageSize=100'
response = requests.get(url)
news = json.loads(response.text)
news=json.dumps(news["articles"], indent=4)

from io import StringIO

df = pd.read_json(StringIO(news))
df.to_csv("output.csv")
datat = pd.read_csv('output.csv')
datat = datat.sample(frac = 1)

tt = datat['title'].fillna('')

finale = datat['title']
finale=pd.DataFrame(finale)
finale.insert(1,"url",datat['url'],True)
finale.insert(1,"date",datat['publishedAt'],True)
finale.insert(1,"desc",datat['description'],True)
max_size = 50
datat=datat.fillna("nil")

datat['description'] = datat['description'].str.split(n=max_size).str[:max_size].str.join(' ')

datat['description']= datat['description'].apply(lambda x: [x])

datat['title'] = datat['title'].apply(lambda x: [x])
k=[]

for index, row in datat.iterrows():
    print(row['description'][0])
    if row['title'] and row['description']:
        k.append(test_similarity(row['title'],row['description']))
    else:
        k.append(0)
print(len(k))
p=model.predict(pd.Series(k))
farray=[]
for i in p:
    i = list(i)
    farray.append(i.index(max(i)))

print(len(farray))
mn = pd.DataFrame(farray)
finale=pd.DataFrame(finale)
finale.insert(1,"OP",mn,True)


finale.to_csv('finale.csv')

# finale.to_json(path_or_buf='finale.json',orient='records')
finale=finale.to_json()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/time')
def get_current_time():
    return {"finale":[finale]}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 8080))