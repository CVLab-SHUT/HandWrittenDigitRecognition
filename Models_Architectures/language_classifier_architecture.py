import cv2
import numpy as np
from mramodule import MRAModule  # Importing the MRAModule
from keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras import Model

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Define the convolutional layer
        self.conv_layer = Conv2D(64 , (3, 3), activation='relu', padding='same')

        # Define the dense layers
        self.dense_layer1 = Dense(512, activation = 'relu')
        self.dense_layer2 = Dense(256, activation = 'relu')
        self.dense_layer3 = Dense(128, activation = 'relu')
        self.output_layer = Dense(10, activation = 'softmax')

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
        x = self.conv_layer(inputs)
        x = self.mra_module1(x)
        x = self.mra_module2(x)
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
        x = tf.keras.Input(shape=(64, 64, 1))
        return Model(inputs=[x],outputs=self.call(x))

if __name__ == "__main__":
    # Instantiate the model
    model = CustomModel()
