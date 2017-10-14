
# coding: utf-8

# In[1]:

import numpy as np

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[2]:

class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x,                 test_y, embedding_layer=None,                 dict_size=5000, example_length=500,                 embedding_length=128, epochs=15, batch_size=32):
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
        if embedding_layer == None:
            model.add(Embedding(dict_size, embedding_length, input_length=example_length))
        else:
            model.add(embedding_layer)
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        print ("dict_size={0}, example_length={1}, embedding_length={2},  batch_size={3}, epochs={4}".format(\
            dict_size, example_length, embedding_length,  batch_size, epochs))
        self.model = model


    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.train_x, self.train_y,                validation_data=(self.test_x, self.test_y),                       epochs=self.epochs, batch_size=self.batch_size)


    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


# In[ ]:




# In[3]:

dict_size=5000
example_length=500
embedding_length=100
epochs=50
batch_size=128


# In[4]:

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=dict_size)


# In[5]:

import os


GLOVE_DIR = "/home/ubuntu/data/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[6]:

EMBEDDING_DIM=embedding_length


# In[7]:


word_index = imdb.get_word_index()


# In[8]:

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[ ]:

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=example_length,
                            trainable=False)


# In[ ]:


(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=dict_size)



rnn = RNN(train_x, train_y, test_x, test_y, embedding_layer = embedding_layer,         dict_size=dict_size, embedding_length=embedding_length, example_length=example_length,             epochs=epochs, batch_size=batch_size)
rnn.train()
rnn.evaluate()


# In[ ]:



