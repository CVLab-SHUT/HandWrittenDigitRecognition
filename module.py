import tensorflow as tf
from tensorflow.keras.layers import  (Reshape,  GlobalAveragePooling2D,  Multiply,
                                      Conv2D,  BatchNormalization)

class Module:
    def __init__(self, inputs, n_neurons, coef):
        self.inputs = inputs
        self.n_neurons = n_neurons
        self.coef = coef
    
    
    def inception_block(self):
        # 1x1 convolution
        conv1 = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (1, 1), activation='relu', padding='same')(self.inputs)
        # 3x3 convolution
        conv3 = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (3, 3), activation='relu', padding='same')(self.inputs)
        # 5x5 convolution
        conv5 = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (5, 5), activation='relu', padding='same')(self.inputs)
        # Add function
        output = tf.keras.layers.add([conv1, conv3, conv5])
        return output
    
    
    def self_attention_block(self, inputs):
        conv = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (3, 3), activation='relu', padding='same')(inputs)
        layer_1 = tf.keras.layers.GlobalAveragePooling2D()(conv)
        layer_2 = tf.keras.layers.Reshape((1, 1, self.n_neurons * self.coef))(layer_1)  # Reshape to add additional dimensions
        layer_3 = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (1, 1), activation='relu')(layer_2)
        layer_4 = tf.keras.layers.BatchNormalization()(layer_3)
        layer_5 = tf.keras.layers.Conv2D(self.n_neurons * self.coef, (1, 1), activation='sigmoid')(layer_4)
        attended_conv = tf.keras.layers.Multiply()([conv, layer_5])
        return attended_conv
    
    
    def call(self):
        inception_output = self.inception_block()
        attention_output = self.self_attention_block(inception_output)
        return attention_output
    
    
if __name__ == "__main__":
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = Module(inputs, 64, 1)
    print(x.call())