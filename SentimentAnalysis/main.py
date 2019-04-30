# -*- coding: utf-8 -*-
"""
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue Apr 30 19:46:23 2019
"""

#import os
#os.chdir('C:/Users/sunbl/Desktop/edata/experiences/8.news/NewsBigDataAnalysis/SentimentAnalysis')
from LoadData import LoadData
from Preprocessing import Preprocessing, EDA
from Model import Model

if __name__ == "__main__":

    # load data. please change the working directory to suit your case
    ld = LoadData('C:/users/sunbl/desktop/')
    train_data = ld.read_data('ratings_train.txt')
    test_data = ld.read_data('ratings_test.txt')
    
    # preprocess the loaded data
    pp = Preprocessing()
    doc1, text1 = pp.tokenize2(train_data[0:200])
    doc2, text2 = pp.tokenize2(test_data[0:200])
    
    # 시각화
    vis = EDA(text1)
    vis.show_freq(50)
    
    # train - test dataset 만들기
    x_train = pp.make_x(doc1, text1) # these are dependent variables
    x_test = pp.make_y(doc1) # these are independent variables
    
    y_train = pp.make_x(doc2, text2) # the same as above
    y_test = pp.make_y(doc2)
    
    mdl = Model()
    model = mdl.my_generate() # build up the model architecture
    model = mdl.my_train(model, x_train, x_test)
    result = mdl.my_eval(model, y_train, y_test)
    
    # test with samples. you can go with your own one
    mdl.predict_pos_neg(model,"올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.",doc1)
    mdl.predict_pos_neg(model,"배경 음악이 영화의 분위기랑 너무 안 맞았습니다. 몰입에 방해가 됩니다.",doc1)
    mdl.predict_pos_neg(model,"주연 배우가 신인인데 연기를 진짜 잘 하네요. 몰입감 ㅎㄷㄷ",doc1)
    mdl.predict_pos_neg(model,"믿고 보는 감독이지만 이번에는 아니네요",doc1)
    mdl.predict_pos_neg(model,"주연배우 때문에 봤어요",doc1)