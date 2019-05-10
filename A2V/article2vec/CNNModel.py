from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.models import Model
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras import optimizers

class CNNModel:
    def train_model(self,num_classes,lr,input_size,dimension):
        title = Sequential()
        contents = Sequential()
        # title
        title.add(Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=[input_size[0],dimension,1]))
        title.add(Conv2D(32, (3, 3), activation='relu'))
        title.add(MaxPooling2D(pool_size=(2, 2)))
        title.add(Dropout(0.25))
        title.add(Flatten())
        # contents
        contents.add(Conv2D(16, kernel_size=(3, 3), activation='relu',input_shape=[input_size[1],dimension,1]))
        contents.add(Conv2D(32, (3, 3), activation='relu'))
        contents.add(MaxPooling2D(pool_size=(2, 2)))
        contents.add(Dropout(0.25))
        contents.add(Flatten())
        # merge
        merged = Concatenate(axis=-1)([title.output,contents.output])
        merged = Dense(256,activation='relu')(merged)   # 사용법 1
        merged = Dense(num_classes,activation="softmax")(merged)
        self.model = Model([title.input, contents.input], merged)

        # https://stackoverflow.com/questions/53560767/valueerror-layer-concatenate-1-was-called-with-an-input-that-isnt-a-symbolic-t
        # self.model = Sequential()
        # self.model.add(Dense(256,activation='relu'))
        # self.model.add(Dense(num_classes,activation='softmax'))
        # self.model.add(merged)
        # self.model =Model([title.input,contents.input],self.model)

        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()