import pickle
import cv2
from keras.utils import np_utils
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16 
import tensorflow as tf
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Input

# image data generator and related functions
# upsample and preprocessing
# vgg base model things
class my_vgg:
    def __init__(self):
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        for layer in self.base_model.layers:
            layer.trainable = False
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(10, activation='softmax')
        self.dropout = Dropout(0.5)
        self.input = Input()
        
    def call(self):
        x = self.base_model(self.input)
        x = self.Flatten()(x)
        x = self.dense1()(x)
        x = self.dropout()(x)
        x = self.dense2()(x)
        return