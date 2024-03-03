from keras.layers import (Conv2D, Dropout, BatchNormalization,
                          MaxPooling2D, Conv2DTranspose, concatenate)
import tensorflow as tf

# This class defines an encoder block for the U-Net architecture.
class UnetEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(UnetEncoderBlock, self).__init__()
        # Define the layers in the block
        self.conv_layer1 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.conv_layer2 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.batch_norm = BatchNormalization()
        self.pooling_layer = MaxPooling2D((2, 2))
        self.dropout_layer = Dropout(0.5)

    def call(self, inputs):
        # Define the forward pass
        conv_output1 = self.conv_layer1(inputs)
        conv_output2 = self.conv_layer2(conv_output1)
        batch_norm_output = self.batch_norm(conv_output2)
        pooling_output = self.pooling_layer(batch_norm_output)
        dropout_output = self.dropout_layer(pooling_output)
        return batch_norm_output, dropout_output

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config

# This class defines a decoder block for the U-Net architecture.
class UnetDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(UnetDecoderBlock, self).__init__()
        # Define the layers in the block
        self.transpose_conv_layer = Conv2DTranspose(num_filters, (3, 3), strides= (2,2),padding='same')
        self.dropout_layer = Dropout(0.5)
        self.conv_layer1 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.conv_layer2 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.batch_norm = BatchNormalization()

    def call(self, inputs):
        # Define the forward pass
        bottleneck_output, mid_res_output = inputs
        transpose_conv_output = self.transpose_conv_layer(bottleneck_output)
        concat_output = concatenate([transpose_conv_output, mid_res_output])
        dropout_output = self.dropout_layer(concat_output)
        conv_output1 = self.conv_layer1(dropout_output)
        conv_output2 = self.conv_layer2(conv_output1)
        batch_norm_output = self.batch_norm(conv_output2)
        return batch_norm_output

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config
   
# This class defines a bottleneck block for the U-Net architecture.
class UnetBottleneck(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(UnetBottleneck, self).__init__()
        # Define the layers in the block
        self.conv_layer1 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.conv_layer2 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')
        self.batch_norm = BatchNormalization()

    def call(self, inputs):
        # Define the forward pass
        conv_output1 = self.conv_layer1(inputs)
        conv_output2 = self.conv_layer2(conv_output1)
        batch_norm_output = self.batch_norm(conv_output2)
        return batch_norm_output

    def get_config(self):
        # Method to support model saving and loading
        config = super().get_config()
        return config
