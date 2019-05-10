# -*- coding: utf-8 -*-

"""
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue Apr 30 19:46:23 2019
"""

import numpy as np
from konlpy.tag import Okt # recommend you to install latest version
import nltk
from keras import models, layers, optimizers, losses, metrics

class Model:
    def __init__(self):
        '''
        @const COMMON_NUM: number of vocabs to care. if it is large, it takes long time to train
        '''
        self.COMMON_NUM = 100

    def my_generate(self):
        '''Function to build up the model architecture
        you can change the number of layers or any regarding parameters of the architecture
        '''
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(self.COMMON_NUM,))) 
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])
        return model

    def my_train(self,model,x_train,x_test):
        '''Function to train the model
        '''
        model.fit(x_train, x_test, epochs=10, batch_size=512)
        return model
    
    def my_eval(self,model,y_train,y_test):
        '''Function to evaluate the model
        '''
        return model.evaluate(y_train, y_test)

    def predict_pos_neg(self,model,review,doc):
        '''Function to predict whether the review is positive / negative
        '''
        okt = Okt()
        token = ['/'.join(t) for t in okt.pos(review, norm=True, stem=True)] # tag is added with the word

        tokens = [t for d in doc for t in d[0]]
        text = nltk.Text(tokens, name='NMSC')

        selected_words = []
        for i in range(len(text.vocab().most_common(100))):
            selected_words.append(text.vocab().most_common(100)[i][0])

        tf = [token.count(word) for word in selected_words]
        data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
        score = float(model.predict(data))
        if(score > 0.5):
            print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
        else:
            print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
