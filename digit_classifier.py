import tensorflow as tf
from keras.layers import (InputLayer, Conv2D, BatchNormalization,
                          Dropout, MaxPooling2D, Dense, Flatten)
from module import Module
from tf.keras.models import load_model
from classifier_model import ClassifierBase

class DigitClassifier(ClassifierBase):
    def __init__(self):
        self.pre_trained_model = load_model("pre_trained_model.h5")
        # extract the existing head
        self.pre_trained_model = self.pre_trained_model.layers[-3:]
        # freeze the model
        for layer in self.pre_trained_model:
            layer.trainable = False
        self.new_hearder= self.head()
        
        
    def head(self, inputs):
        dense = Dense(256, activation = 'relu')(inputs)
        dense = Dropout(0.25)(dense)
        dense = Dense(128, activation = 'relu')(dense)
        dense = Dropout(0.25)(dense)
        dense = Dense(10, activation = 'softmax')(dense)
        return dense
        
        
    def call(self, inputs):
        x = self.pre_trained_model(inputs)
        x = self.new_hearder(x)
        return x
        
        