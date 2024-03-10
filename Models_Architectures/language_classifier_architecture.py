import cv2
import numpy as np
from mramodule import MRAModule  # Importing the MRAModule
from keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.keras import Model

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Define the convolutional layer
        self.conv_layer1 = Conv2D(64 , (3, 3), activation='relu', padding='same')
        self.conv_layer2 = Conv2D(64 , (3, 3), activation='relu', padding='same')
        self.conv_layer3 = Conv2D(128 , (3, 3), activation='relu', padding='same')
        self.conv_layer4 = Conv2D(128 , (3, 3), activation='relu', padding='same')
        self.conv_layer5 = Conv2D(256 , (3, 3), activation='relu', padding='same')
        self.conv_layer6 = Conv2D(256 , (3, 3), activation='relu', padding='same')
        self.maxPooling = MaxPool2D(2, 2)
        # Define the dense layers
        self.dense_layer1 = Dense(512, activation = 'relu')
        self.dense_layer2 = Dense(256, activation = 'relu')
        self.dense_layer3 = Dense(128, activation = 'relu')
        self.output_layer = Dense(12, activation = 'softmax')

        # Define the flatten layer
        self.flatten_layer = Flatten()

        # Define the dropout layers
        self.dropout_layer1 = Dropout(0.25)
        self.dropout_layer2 = Dropout(0.25)
        self.dropout_layer3 = Dropout(0.25)

        # Define the MRAModules
        self.mra_module1 = MRAModule()
        self.mra_module2 = MRAModule()
        self.mra_module3 = MRAModule()

    def call(self, inputs):
        # Pass the inputs through the layers
        x = self.conv_layer1(inputs)
        x = self.conv_layer2(x)
        x = self.maxPooling(x)
        x = self.mra_module1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.maxPooling(x)
        x = self.mra_module2(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.maxPooling(x)
        x = self.mra_module3(x)
        x = self.flatten_layer(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer1(x)
        x = self.dense_layer2(x)
        x = self.dropout_layer2(x)
        x = self.dense_layer3(x)
        x = self.dropout_layer3(x)
        x = self.output_layer(x)
        return x

    def model(self):
        # Define the input shape
        x = tf.keras.Input(shape=(128, 128, 1))
        return Model(inputs=[x],outputs=self.call(x))

if __name__ == "__main__":
    # Instantiate the model
    model = CustomModel()
