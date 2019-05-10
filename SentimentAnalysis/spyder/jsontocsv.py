# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:29:49 2019

@author: sunbl
"""

import pandas as pd
import json
import os

os.chdir('C:/users/sunbl/desktop/edata/experiences/7.news/dta/tweet')

def getdta(filename, word):
    tweets = []
    for line in open(filename, 'r', encoding='utf-8-sig'):
        tweets.append(json.loads(line))
    df = pd.DataFrame.from_dict(tweets, orient='columns')
    x = df[df['text'].str.contains(word)]
    return x['text']

dta = getdta('twitter_20190417.json', '유천')
dta.to_csv("417.csv")


tweets = []
for line in open('twitter_20190417.json', 'r', encoding='utf-8-sig'):
    tweets.append(json.loads(line))
df = pd.DataFrame.from_dict(tweets, orient='columns')
