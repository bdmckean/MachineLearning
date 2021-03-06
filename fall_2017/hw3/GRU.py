
# coding: utf-8

# In[1]:

import numpy as np
import datetime

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1DM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[ ]:

class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=5000, example_length=500, embedding_length=32, epochs=15, batch_size=128):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.example_len = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length

        # TODO:preprocess training data
        self.train_x = sequence.pad_sequences(train_x, maxlen=example_length)
        self.test_x = sequence.pad_sequences(test_x, maxlen=example_length)
        self.train_y = train_y
        self.test_y = test_y

        # TODO:build model
        # create the model
        model = Sequential()
        model.add(Embedding(dict_size, embedding_length, input_length=example_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model


    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.train_x, self.train_y,\
            validation_data=(self.test_x, self.test_y),\
                epochs=self.epochs, batch_size=self.batch_size)


    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size)


# In[ ]:

time = datetime.datetime.now()
print (time )
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
rnn = RNN(train_x, train_y, test_x, test_y, example_length=500, batch_size=512, epochs=15)
rnn.train()
rnn.evaluate()

print (datetime.datetime.now() - time)


# In[ ]:



