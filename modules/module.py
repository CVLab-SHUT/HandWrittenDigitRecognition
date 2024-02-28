import tensorflow as tf
from tensorflow.keras.layers import Reshape, GlobalAveragePooling2D, Multiply, Conv2D, BatchNormalization
from tensorflow.keras import Model

class Module(Model):
    def __init__(self, inputs):
        super(Module, self).__init__()
        self.inputs = inputs
        
    def inception_block(self, inputs):
        conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv5 = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
        output = tf.keras.layers.add([conv1, conv3, conv5])
        return output

    def self_attention_block(self, inputs):
        conv = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        layer_1 = GlobalAveragePooling2D()(conv)
        layer_2 = Reshape((1, 1, 64))(layer_1)
        layer_3 = Conv2D(64, (1, 1), activation='relu')(layer_2)
        layer_4 = BatchNormalization()(layer_3)
        layer_5 = Conv2D(64, (1, 1), activation='sigmoid')(layer_4)
        attended_conv = Multiply()([conv, layer_5])
        return attended_conv

    # def __call__(self):
    #     inception_output = self.inception_block(self.inputs)
    #     attention_output = self.self_attention_block(inception_output)
    #     return attention_output

if __name__ == "__main__":
    inputs = tf.keras.Input(shape=(224, 224, 3))
    model = Module(inputs)