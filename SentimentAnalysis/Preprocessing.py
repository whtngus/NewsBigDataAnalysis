# -*- coding: utf-8 -*-

'''
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue Apr 30 01:26:01 2019
'''

from konlpy.tag import Okt # recommend you to install latest version
from pprint import pprint
import nltk
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np


class Preprocessing:
    def __init__(self):
        '''
        @const COMMON_NUM: number of vocabs. you can change it but SHOULD BE IDENTICAL to the one in the Model.py
        '''
        self.COMMON_NUM = 100

    def tokenize(self,data):
        okt = Okt()
        return ['/'.join(t) for t in okt.pos(data, norm=True, stem=True)] # norm은 정규화, stem은 근어(stemming)

    def tokenize2(self,data):
        doc = [(self.tokenize(row[1]), row[2]) for row in data]
        tok = [t for d in doc for t in d[0]]
        text = nltk.Text(tok, name='NMSC') # these are the tokens (ex. '그리다/Verb')
        return doc, text;

    def print_info(self,text,n):
        print("Number of tokens: ", len(text.tokens)) # Total number of tokens
        print("Without redundancy: ", len(set(text.tokens))) # 중복을 제외한 토큰의 개수
        pprint(text.vocab().most_common(n)) # 출현 빈도가 높은 상위 토큰 n개
        
    def selected_words(self,text):
        # 시간이 꽤 걸립니다! 시간을 절약하고 싶으면 most_common의 매개변수를 줄여보세요.
        return [f[0] for f in text.vocab().most_common(self.COMMON_NUM)]

    def term_frequency(self,doc,text):
        return [doc.count(word) for word in self.selected_words(text)]

    def make_x(self,doc,text):
        x = [self.term_frequency(d, text) for d, _ in doc]
        return np.asarray(x).astype('float32')

    def make_y(self,doc):
        y = [c for _, c in doc]
        return np.asarray(y).astype('float32')

class EDA:
    def __init__(self,text):
        '''
        @param text: text returned from Preprocessing.tokenzie2
        '''
        self.text = text

    def show_freq(self,n):
        '''Function to see the frequency of the words.
        @param n: number of words
        '''
        #%matplotlib inline
        font_fname='c:/windows/fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname=font_fname).get_name()
        rc('font', family=font_name)
        plt.figure(figsize=(20,10))
        self.text.plot(n)
