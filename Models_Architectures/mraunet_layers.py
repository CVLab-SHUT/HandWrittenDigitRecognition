from keras.layers import (Conv2D, Dropout, BatchNormalization,
                          MaxPooling2D, Conv2DTranspose, concatenate)
import tensorflow as tf
from keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D

# This class defines an encoder block for the U-Net architecture.
class UnetEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_number):
        super(UnetEncoderBlock, self).__init__()
        # Define the layers in the block
        self.conv1 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))
        self.do1 = Dropout(0.5)

    def call(self, inputs):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.conv2(x)
        x1 = self.bn1(x)
        x2 = self.pool1(x1)
        x2 = self.do1(x2)
        return x1, x2

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config

# This class defines a decoder block for the U-Net architecture.
class UnetDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_number):
        super(UnetDecoderBlock, self).__init__()
        # Define the layers in the block
        self.convT1 = Conv2DTranspose(n_number, (3, 3), strides= (2,2),padding='same')
        self.do1 = Dropout(0.5)
        self.conv1 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.bn1 = BatchNormalization()

    def call(self, inputs):
        # Define the forward pass
        bneck, mra = inputs
        x = self.convT1(bneck)
        x = concatenate([x, mra])
        x = self.do1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        return x

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config
   
# This class defines a bottleneck block for the U-Net architecture.
class UnetBottleneck(tf.keras.layers.Layer):
    def __init__(self, n_number):
        super(UnetBottleneck, self).__init__()
        # Define the layers in the block
        self.conv1 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.conv2 = Conv2D(n_number, (3, 3), activation='relu', padding='same')
        self.bn1 = BatchNormalization()

    def call(self, inputs):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        return x

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config
