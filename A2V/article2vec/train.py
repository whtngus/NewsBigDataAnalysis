from article2vec.DataLoader import DataLoader
from keras.models import load_model
from article2vec import GRUModel
import numpy as np

class ClassificationTrain:

    def __init__(self, trainPath,modelPath):
        '''

        :param trainPath : 대상파일 경로
        :param modelPath 모댈저장 경로
        '''
        self.trainPath = trainPath
        self.modelPath = modelPath
        self.data_loader = DataLoader()

    def train(self):
        '''
        train
        :return:
        '''
        sign_onehot = "sign_onehot"
        input_size = 20
        step_size = 5
        batch_size= 2000
        data_list, label_list = self.data_loader.train_data_loader(self.trainPath,sign_onehot)
        train_data,label_data,sig_size,_ = self.data_loader.data_embedding(data_list,label_list,sign_onehot,input_size,step_size)

        model_gru = GRUModel.GRUModel()
        model_gru.train_model(sig_size,input_size)
        model = model_gru.model
        train_history = model.fit(train_data, label_data, epochs=20, batch_size=batch_size,verbose=2)
        train_history = train_history.history
        model.save(self.modelPath)

    def test(self):
        '''
            test ㅠㅠ
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
        train = ClassificationTrain("../data/newsData.csv", "model5")
        train.test()
    else:
        train = ClassificationTrain("train", "model2")
        train.train()