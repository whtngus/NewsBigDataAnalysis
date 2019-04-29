from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Flatten
from keras import optimizers

class GRUModel:

    def train_model(self,sig_size,input_size):
        self.model = Sequential()

        self.model.add(GRU(20, return_sequences=True, input_shape=(input_size, sig_size+1),activation='tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(GRU(3, return_sequences=False,activation='tanh'))
        self.model.add(Dropout(0.5))
        # self.model.add(Flatten())
        self.model.add(Dense(units=3,activation='sigmoid'))
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()