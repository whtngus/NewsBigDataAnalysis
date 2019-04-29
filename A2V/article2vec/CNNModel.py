from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Flatten
from keras import optimizers

class CNNModel:
    def train_model(self,num_classes,lr):
        self.model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()