import tensorflow as tf
from keras.layers import (InputLayer, Conv2D,BatchNormalization,
                          Dropout,MaxPooling2D, Dense, Flatten)
from module import Module
from classifier_model import ClassifierBase

class LanguageClassifier(ClassifierBase):
    def __init__(self, i_shape, n_neurons = 64, coef = 1, num_classes = 12):
        super(LanguageClassifier, self).__init__()
        self.input = InputLayer(input_shape = i_shape)
        self.n_neurons = n_neurons
        self.coef = coef
        self.block = self.conv_block()
        self.header = self.head()
        
        
    def conv_block(self, inputs):
        conv = Conv2D(self.n_neurons , (3, 3), activation='relu', padding='same')(inputs)
        conv = BatchNormalization()(conv)
        conv = Conv2D(self.n_neurons , (3, 3), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPooling2D((2, 2))(conv)
        conv = Module(conv, self.n_neurons, self.coef).call()
        return conv
        
        
    def head(self, inputs):
        flatten = Flatten()(inputs)
        dense = Dense(512, activation = 'relu')(flatten)
        dense = Dropout(0.25)(dense)
        dense = Dense(256, activation = 'relu')(dense)
        dense = Dropout(0.25)(dense)
        dense = Dense(128, activation = 'relu')(dense)
        dense = Dropout(0.25)(dense)
        dense = Dense(10, activation = 'softmax')(dense)
        return dense


    # overriding call function
    def call(self):
        x = self.block(self.input)
        x = self.block(x)
        x = self.block(x)
        x = self.head(x)
        return x
    
        
if __name__ == "__main__":
    classifier = LanguageClassifier((128,128,1), n_neurons = 64, coef = 1, num_classes = 12)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # classifier.fit(train_data, epochs=10, validation_data=validation_data)
