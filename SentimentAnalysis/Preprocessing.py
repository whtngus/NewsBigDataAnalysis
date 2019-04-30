# -*- coding: utf-8 -*-

'''
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue Apr 30 01:26:01 2019
'''

from konlpy.tag import Okt # recommend you to install latest version
import os
from pprint import pprint
import nltk
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

class Preprocessing:
    def tokenize(self,data):
        okt = Okt()
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(data, norm=True, stem=True)]

    def tokenize2(self,data):
        doc = [(self.tokenize(row[1]), row[2]) for row in data]
        tok = [t for d in doc for t in d[0]]
        text = nltk.Text(tok, name='NMSC')
        return text # this is the tokens (ex. '그리다/Verb')

    def printInfo(self,text,n):
        print("Number of tokens: ", len(text.tokens)) # Total number of tokens
        print("Without redundancy: ", len(set(text.tokens))) # 중복을 제외한 토큰의 개수
        pprint(text.vocab().most_common(n)) # 출현 빈도가 높은 상위 토큰 n개

class EDA: 







test_data = read_data('ratings_test.txt')
x = Preprocessing()
z = x.tokenize2(test_data[0:10])
x.printInfo(z,3)





%matplotlib inline

font_fname = 'c:/windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)

plt.figure(figsize=(20,10))
text.plot(50)

rc('font', family=font_name)


COMMON_NUM = 100


# 시간이 꽤 걸립니다! 시간을 절약하고 싶으면 most_common의 매개변수를 줄여보세요.
selected_words = [f[0] for f in text.vocab().most_common(COMMON_NUM)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]


import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


