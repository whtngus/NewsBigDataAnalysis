# -*- coding: utf_8 -*-
import numpy as np
from random import shuffle
import sys
import csv
from konlpy.tag import Twitter

class DataLoader:
    def __init__(self,label_count,input_size):
        # self.fast_text = self.load_fest_text()  # 데이터가 커서 임시 주석
        self.label_count = label_count
        self.input_size = input_size
        self.dimension = 300
        self.spliter = Twitter()

    def load_fest_text(self):
        fast_text = {}
        try:
            with open("../data/cc.ko.300.vec", "r", encoding="utf-8") as target:
                target.readline()
                for line in target:
                    data = line.split()
                    fast_text[data[0]] = [float(vec) for vec in data[1:]]
        except Exception as e:
            print("word2Vec load faile : ",e)
        return fast_text

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
        return self.input_embedding(data_list),self.label_embedding(label_list)

    def input_embedding(self,data_list):
        for data_index, data in enumerate(data_list):
            word_index = 0
            data_list[data_index] = [self.w2v_match(data[0],self.input_size[0]),self.w2v_match(data[1],self.input_size[1])]
        return data_list

    def w2v_match(self,sentence,size):
        input_shape = [[0] * self.dimension] * size
        word_index = 0
        for word in self.spliter.pos(sentence):
            if word[1] in ['Noun','Verb'] and word[0] in self.fast_text:
                input_shape[word_index] = self.fast_text[word[0]][:]
                word_index += 1
        return input_shape


    def label_embedding(self,label_list):
        # 테그를 one-hot 으로 치환시키기 위해 사용
        one_hot_label = np.eye(self.label_count)
        for rb_index in range(len(label_list)):
            label_list[rb_index] = one_hot_label[int(label_list[rb_index]) - 1][:]
        return label_list


if __name__ == "__main__":
    label_count = 3
    input_size = [20,5]
    dataPath = "../data/test2.csv"
    dataLoader = DataLoader(label_count,input_size)
    train_data, test_data = dataLoader.data_loader(dataPath)