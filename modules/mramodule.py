from tensorflow.keras.layers import Multiply, Conv2D, GlobalAveragePooling2D, Reshape, BatchNormalization, add
import tensorflow as tf

class MRAModule(tf.keras.layers.Layer):
    """
    This class defines a Multi-Resolution Attention (MRA) module.
    """
    def __init__(self):
        super(MRAModule, self).__init__()

        # Define the layers in the module
        # Convolutional layers with different kernel sizes
        self.conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(64, (5, 5), activation='relu', padding='same')

        # Convolutional layers for the attention mechanism
        self.conv4 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv5 = Conv2D(64, (1, 1), activation='relu', padding='same')
        self.conv6 = Conv2D(64, (1, 1), activation='relu', padding='same')

        # Batch normalization layer
        self.bn1 = BatchNormalization()

        # Global average pooling layer
        self.gap = GlobalAveragePooling2D()

    def call(self, inputs):
        """
        Define the forward pass.
        """
        # Apply convolutional layers
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(inputs)

        # Combine the outputs
        x = add([conv1, conv2, conv3])

        # Apply the attention mechanism
        conv4 = self.conv4(x)
        x = self.gap(conv4)
        x = Reshape((1, 1, 64))(x)
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.conv6(x)

        # Multiply the original feature map with the attention map
        output = Multiply()([conv4, x])

        return output

    def get_config(self):
        """
        Method to support model saving and loading.
        """
        config = super().get_config()
        return config
