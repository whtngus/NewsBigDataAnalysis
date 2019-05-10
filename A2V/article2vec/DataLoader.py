# -*- coding: utf_8 -*-
import numpy as np
from random import shuffle
import sys
import csv
from konlpy.tag import Twitter

class DataLoader:
    def __init__(self,label_count,input_size,train_rate,dimension):
        '''
        :param label_count: 정답 분류 갯수
        :param input_size: 본문, 제목 에서 뽑을 단어 수
        :param train_rate: 트레이닝 테스트 나눌 비율
        :param dimension: 각 단어당 매칭될 벡터 크기
        '''
        # self.word2vec = self.load_fest_text()
        self.word2vec = self.load_word2vec_30()
        self.dimension = dimension
        self.label_count = label_count
        self.input_size = input_size
        self.spliter = Twitter()
        self.train_rate = train_rate

    def load_word2vec_30(self):
        word2vec = {}
        try:
            with open("../data/ko.tsv", "r", encoding="utf-8") as target:
                count = 0
                surface = ""
                for line in target:
                    data = line.split()
                    if data[0] == str(count):
                        surface = data[1]
                        word2vec[surface] =[float(vec) for vec in data[2:]]
                        count+=1
                    else:
                        word2vec[surface] = word2vec[surface] + [float(vec) for vec in data]
        except Exception as e:
            print("word2Vec load faile : ",e)
        return word2vec

    def load_fest_text(self):
        word2vec = {}
        try:
            with open("../data/cc.ko.300.vec", "r", encoding="utf-8") as target:
                target.readline()
                for line in target:
                    data = line.split()
                    word2vec[data[0]] = [float(vec) for vec in data[1:]]
        except Exception as e:
            print("word2Vec load faile : ",e)
        return word2vec

    def data_divide_write(self,data_path,output_path1,output_path2):
        lines = []
        try:
            with open(data_path, "r", encoding="utf-8") as target:
                for line in target:
                    lines.append(line)
        except FileNotFoundError as e:
            print("해당 파일이 존재하지 않습니다.")
            sys.exit(1)
        shuffle(lines)
        train_size = int((len(lines)/10) * 9)
        with open(output_path1, "w",encoding="utf-8") as f:
            for index in range(train_size):
                f.write(lines[index])

        with open(output_path2, "w", encoding="utf-8") as f:
            for index in range(len(lines)-train_size):
                f.write(lines[index+train_size])

    def data_loader(self,data_path):
        '''
        대상파일을 읽어들임
        @:param data_list 읽어들일 데이터
        @:param onehot_path sign onehot 매칭할 데이터
        return 읽어들인 data, label
        '''
        label_list = []
        data_list = []
        try:
            with open(data_path, "r", encoding="utf-8") as target:
                csv_file = csv.reader(target)
                for line in csv_file:
                    label_list.append(line[2])
                    data_list.append([line[0],line[1]])
        except FileNotFoundError as e:
            print("해당 파일이 존재하지 않습니다.")
            sys.exit(1)
        rate = int(len(data_list) *self.train_rate)
        input = self.input_embedding(data_list)
        label = self.label_embedding(label_list)
        train_input = input[:rate]
        train_len = len(train_input)
        train_input1 =  np.array([i for i,j in train_input]).reshape(train_len,self.input_size[0],self.dimension,1)
        train_input2 =  np.array([j for i, j in train_input]).reshape(train_len,self.input_size[1],self.dimension,1)
        train_label = np.array(label[:rate])
        test_input = input[rate:]
        test_len = len(test_input)
        test_input1 =   np.array([i for i,j in test_input]).reshape(test_len,self.input_size[0],self.dimension,1)
        test_input2 =  np.array([j for i, j in test_input]).reshape(test_len,self.input_size[1],self.dimension,1)
        test_label = np.array(label[rate:])
        return train_input1, train_input2, train_label, test_input1, test_input2, test_label

    def input_embedding(self,data_list):
        change_data_list = []
        for data in data_list:
            change_data_list.append([self.w2v_match(data[0],self.input_size[0]),self.w2v_match(data[1],self.input_size[1])])
        return change_data_list

    def w2v_match(self,sentence,size):
        input_shape = [[0.0] * self.dimension] * size
        word_index = 0
        for word in self.spliter.pos(sentence):
            if word[1] in ['Noun','Verb'] and word[0] in self.word2vec:
                input_shape[word_index] = self.word2vec[word[0]][:]
                word_index += 1
            if word_index >= size:
                break
        return input_shape


    def label_embedding(self,label_list):
        # 테그를 one-hot 으로 치환시키기 위해 사용
        one_hot_label = np.eye(self.label_count)
        for rb_index in range(len(label_list)):
            label_list[rb_index] = one_hot_label[int(label_list[rb_index]) - 1][:]
        return label_list


if __name__ == "__main__":
    label_count = 3
    train_rate = 0.8
    dimension = 200
    input_size = [30,10]
    dataPath = "../data/test2.csv"
    dataLoader = DataLoader(label_count,input_size,train_rate,dimension)
    train_input, train_label, test_input, test_label = dataLoader.data_loader(dataPath)
    print()