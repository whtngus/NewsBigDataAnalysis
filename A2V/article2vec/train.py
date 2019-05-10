from DataLoader import DataLoader
from keras.models import load_model
from  CNNModel import CNNModel
import numpy as np

class ClassificationTrain:

    def __init__(self, trainPath,modelPath,label_count,input_size,train_rate,dimension):
        self.trainPath = trainPath
        self.modelPath = modelPath
        self.input_size = input_size
        self.dimension = dimension
        self.data_loader = DataLoader(label_count,input_size,train_rate,dimension)

    def train(self,epochs,batch_size,lr):
        '''

        :param epochs: epochs
        :param batch_size: batch Size
        :param lr : learningRate
        :return:
        '''
        train_input1, train_input2, train_label, test_input1, test_input2, test_label = self.data_loader.data_loader(self.trainPath)
        label_count = len(train_label[0])
        model_cnn = CNNModel()
        model_cnn.train_model(label_count, lr,self.input_size,self.dimension)
        model = model_cnn.model
        train_history = model.fit([train_input1,train_input2], train_label, epochs=epochs, batch_size=batch_size,verbose=2
                                  ,validation_data=([test_input1,test_input2],test_label))
        train_history_detail = train_history.history
        model.save(self.modelPath)

    def test(self):
        '''

        :return:
        '''
        sign_onehot = "sign_onehot"
        input_size = 20
        step_size = 5
        # input_size = 10
        # step_size = 2
        data_list, label_list = self.data_loader.test_data_loader(self.trainPath)
        train_data, label_data, sig_size,divide_index = self.data_loader.data_embedding(data_list, label_list, sign_onehot, input_size, step_size)

        model = load_model(self.modelPath)
        lastindex=0
        yhat = model.predict_classes(np.array(train_data))
        y_result = []
        for index in divide_index:
            result = set([])
            for i in range(lastindex,index+1):
                result.add(yhat[i])
                if len(result) >= 3:
                    break
            lastindex = index + 1
            y_result.append(result)

        # recal precision 계산
        tp = 0
        fp = 0
        fn = 0
        for index, hat in enumerate(y_result):
            check = False
            for data in hat:
                if str(data)== label_list[index]:
                    tp +=1
                    check = True
                else:
                    fp += 1
            if check == False:
                fn += 1

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        if precision == 0 and recall ==0:
            print(0)
        else:
            fhalf_score = (1.25 * precision*recall)/(0.25*precision + recall)
            print(fhalf_score)



if __name__ == "__main__":
    mode = "train"
    if mode == "test":
        train = ClassificationTrain("../data/newsData.csv", "model")
        train.test()
    elif mode == "train":
        trainPath = "../data/test2.csv"
        modelPath = "model"
        label_count = 3
        train_rate = 0.8
        input_size = [10, 30]
        dimension = 200
        train = ClassificationTrain(trainPath,modelPath,label_count,input_size,train_rate,dimension)
        epoch = 1000
        batch_size = 10
        lr = 0.01
        train.train(epoch,batch_size,lr)