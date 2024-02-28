import cv2
import numpy as np
from mramodule import MRAModule
from keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Model

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 =  Conv2D(64 , (3, 3), activation='relu', padding='same')
        self.dense1 = Dense(512, activation = 'relu')
        self.dense2 = Dense(256, activation = 'relu')
        self.dense3 = Dense(128, activation = 'relu')
        self.dense4 = Dense(10, activation = 'softmax')
        self.flatten = Flatten()
        self.do1 = Dropout(0.25)
        self.do2 = Dropout(0.25)
        self.do3= Dropout(0.25)
        self.mra1 = MRAModule()
        self.mra2 = MRAModule()
        self.mra3 = MRAModule()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.mra1(x)
        x = self.mra2(x)
        x = self.mra3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.do1(x)
        x = self.dense2(x)
        x = self.do2(x)
        x = self.dense3(x)
        x = self.do3(x)
        x = self.dense4(x)
        return x

    def model(self):
        x = tf.keras.Input(shape=(64, 64, 1))
        return Model(inputs=[x],outputs=self.call(x))

if __name__ == "__main__":
    model = CustomModel()