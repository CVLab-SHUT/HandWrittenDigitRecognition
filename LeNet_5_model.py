import tensorflow as tf
from keras.layers import (InputLayer, Conv2D, AveragePooling2D, Dense, Flatten)
from classifier_base import ClassifierBase
# sth
# LeNet-5 model
class LeNet(ClassifierBase):
    def __init__(self, i_shape, n_classes):
        super(LeNet, self).__init__()
        self.input = InputLayer(input_shape = i_shape)
        self.conv1 = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=i_shape, padding="same")
        self.conv2 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')
        self.flatten = Flatten()
        self.header = self.head()
        self.dense1 = Dense(120, activation='tanh')
        self.dense2 = Dense(84, activation='tanh')
        self.dense3 = Dense(n_classes, activation='softmax')
        self.averagePool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.averagePool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        
    def head(self):
        x = self.flatten(x) 
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
        
    # overriding call function
    def call(self):
        x = self.block1(self.input)
        x = self.conv1(x)
        x = self.averagePool1(x)
        x = self.conv2(x)
        x = self.averagePool2(x)
        x = self.head(x)
        return x