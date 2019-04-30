# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:17:37 2019

@author: sunbl
"""

# -*- coding: utf-8 -*-

'''
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue May 1 03:26:48 2019
'''

# This is for the users who are familiar with the Jupyter Notebook.
# you can run it line by line.


import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


# function to read data
def read_data(filename):
    os.chdir('C:/users/sunbl/desktop/')
    with open(filename, 'r', encoding='UTF8') as f: # set UTF8 since it's Korean
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # remove header
    return data

# separate train / test data
train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

# print out and check the dataset
print(len(train_data))
print(len(train_data[0]))
print(len(test_data))
print(len(test_data[0]))


# import konlpy package. its version should be up-to-date.
# I personally uninstalled and reinstalled it to load Okt.
# I guess it wasn't a part of konlpy package until last Autumn.
from konlpy.tag import Okt

okt = Okt()

# run example code to see it's working well
print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

from pprint import pprint

# tokenize
def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

# tokenize
train_docs = [(tokenize(row[1]), row[2]) for row in train_data[1:100]]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data[1:100]]

# would be like below
# ['흠/Noun', '.../Punctuation', '포스터/Noun','보고/Noun','초딩/Noun','영화/Noun','줄/Noun] '1')

# think of '은' in Korean.
# it can be silver(noun)/ eun(Josa). which should be distinguished
# You see why we should doing this.
pprint(train_docs[0])

# collect tokens
tokens = [t for d in train_docs for t in d[0]]
tokens2 = [t for d in test_docs for t in d[0]]
print(len(tokens))


# import nltk
import nltk
text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))


# visualization,, just fancy miscellaneous work
%matplotlib inline
font_fname = 'c:/windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)
plt.figure(figsize=(20,10))
text.plot(50)


# as mentioned before, you can change the value
COMMON_NUM = 100

# 시간이 꽤 걸립니다! 시간을 절약하고 싶으면 most_common의 매개변수 = COMMON_NUM을 줄여보세요.
selected_words = [f[0] for f in text.vocab().most_common(COMMON_NUM)]

# check the frequency
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

# separate train / test set
train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]


import numpy as np

# change list -> array
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')
y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


# build up Keras model
from keras import models, layers, optimizers, losses, metrics

# architecture
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(100,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# cmopile
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

# train
model.fit(x_train, y_train, epochs=10, batch_size=512)

# see the outcome
results = model.evaluate(x_test, y_test)

# prediction
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
        

# predict with example sentences
predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")
predict_pos_neg("배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.")
predict_pos_neg("주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ")
predict_pos_neg("믿고 보는 감독이지만 이번에는 아니네요")
predict_pos_neg("주연배우 때문에 봤어요")

