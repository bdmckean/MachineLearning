
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
class CNN:
    '''
    CNN classifier
    '''
    #def __init__(self, train_x, train_y, test_x, test_y, history, epochs = 15, batch_size=128, ):
    def __init__(self, train_x, train_y, test_x, test_y, epochs = 15, batch_size=128, ):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epochs = epochs

        print (len(train_x))
        print (len([elem for elem in train_x]))
        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        #self.train_x = np.array(np.array([train_x[i:i + 28] for i in range(0, len(train_x), 28)]))
        #self.train_x = np.array([[elem[i:i + 28] for i in range(0, len(elem), 28)] for elem in train_x])
        #self.test_x = np.array([[elem[i:i + 28] for i in range(0, len(elem), 28)] for elem in test_x])
        self.train_y = np.array(train_y)
        self.test_y = np.array(test_y)
        
        # input image dimensions
        img_x, img_y = 28, 28
        input_shape = (img_x, img_y, 1)
        
        # TODO: reshape train_x and test_x
        self.train_x = train_x.reshape(train_x.shape[0], img_x, img_y, 1)
        self.test_x = test_x.reshape(test_x.shape[0], img_x, img_y, 1)
        

        print (self.train_x.shape, self.test_x.shape, self.train_y.shape, self.test_y.shape)


        # normalize data to range [0, 1]
        #self.train_x /= 255
        #self.test_x /= 255


        # TODO: one hot encoding for train_y and test_y
        num_classes = len(set(train_y))
        one_hot_train_y = [ [0 if elem != idx else 1 for idx in range(num_classes) ] for elem in train_y]
        one_hot_test_y = [ [0 if elem != idx else 1 for idx in range(num_classes) ] for elem in test_y]
        self.train_y = one_hot_train_y
        self.test_y =  one_hot_test_y
        
    
        # TODO: build you CNN model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
        
        self.model = model    


        
    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=1,
          validation_data=(self.test_x, self.test_y))
          #,
          #callbacks=[history])

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    
    #print ( data.train_x.shape, data.test_x.shape, data.train_y.shape, data.test_y.shape )
   
    args.limit = 50000
    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
    #print(cnn.model.summary())
