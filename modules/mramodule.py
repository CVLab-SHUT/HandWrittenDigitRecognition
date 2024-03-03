from keras.layers import Multiply, Conv2D, GlobalAveragePooling2D, Reshape, BatchNormalization, add
import tensorflow as tf

# This class defines a Multi-Resolution Attention (MRA) module.
class MRAModule(tf.keras.layers.Layer):
    def __init__(self):
        super(MRAModule, self).__init__()
        # Define the layers in the module
        self.conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(64, (5, 5), activation='relu', padding='same')
        self.conv4 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv5 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv6 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.bn1 = BatchNormalization()
        self.gap = GlobalAveragePooling2D()

    def call(self, inputs):
        # Define the forward pass
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(inputs)
        x = add([conv1, conv2, conv3])

        conv4 = self.conv4(x)
        x = self.gap(conv4)
        x = Reshape((1, 1, 64))(x)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.conv6(x)
        output = Multiply()([conv4, x])

        return output

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config
