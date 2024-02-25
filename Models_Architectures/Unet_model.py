import tensorflow as tf
from keras.layers import (InputLayer, Conv2D,BatchNormalization,
                          Dropout,MaxPooling2D,Conv2DTranspose,
                          concatenate)
from module import Module

class RA_Unet(tf.keras.Model):
    def __init__(self, i_shape, n_neurons):
        super(RA_Unet, self).__init__()
        self.input = InputLayer(input_shape = i_shape)
        self.n_neurons = n_neurons
        
        self.endcoder_block1 = self.endcoder()
        self.endcoder_block2 = self.endcoder()
        self.endcoder_block3 = self.endcoder()
        self.endcoder_block4 = self.endcoder()
        self.endcoder_block5 = self.endcoder()
        
        self.decoder_block1 = self.decoder()
        self.decoder_block2 = self.decoder()
        self.decoder_block3 = self.decoder()
        self.decoder_block4 = self.decoder()
        self.decoder_block5 = self.decoder()
        
        self.Module1 = Module()
        self.Module2 = Module()
        self.Module3 = Module()
        self.Module4 = Module()
        
        self.bottle_neck_block1 = self.bottle_neck()
        self.bottle_neck_block2 = self.bottle_neck()
        
        self.header = self.head()


    def endcoder(self,input, coef):
        conv = Conv2D(self.n_neurons * coef, (3, 3),
                      activation='relu', padding='same')(input)
        conv = Conv2D(self.n_neurons * coef, (3, 3),
                      activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        pool = MaxPooling2D((2, 2))(conv)
        pool = Dropout(0.5)(pool)
        return  conv, pool
    
    
    def bottle_neck(self,input, coef):
        convm = Conv2D(self.n_neurons * coef, (3, 3),
                       activation='relu', padding='same')(input)
        convm = Conv2D(self.n_neurons * coef, (3, 3),
                       activation='relu', padding='same')(convm)
        convm = BatchNormalization()(convm)
        return convm
    
    
    def decoder(self,input, coef, conv):
        deconv = Conv2DTranspose(self.n_neurons*coef, (3,3),
                                 strides= (2,2), padding='same')(input)
        deconv = concatenate([deconv, conv])
        deconv = Dropout(0.5)(deconv)
        deconv = Conv2D(self.n_neurons * coef, (3, 3),
                        activation='relu', padding='same')(deconv)
        deconv = Conv2D(self.n_neurons * coef, (3, 3),
                        activation='relu', padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        return deconv
    
    
    def head(self, input):
        output_layer = Conv2D(1, (1, 1), padding = 'same',
                              activation = 'sigmoid')(input)
        return output_layer
    
    
    # Overriding call function
    def call(self):
        
        # Encoding Architecture
        en_conv1, pool1 = self.endcoder_block1(self.input, 1)
        en_conv2, pool2 = self.endcoder_block2(pool1, 2)
        en_conv3, pool3 = self.endcoder_block3(pool2, 4)
        en_conv4, pool4 = self.endcoder_block4(pool3, 8)

        # Skip Block
        en_conv1 = self.Module1(en_conv1, 1)
        en_conv2 = self.Module2(en_conv2, 2)
        en_conv3 = self.Module3(en_conv3, 3)
        en_conv4 = self.Module4(en_conv4, 4)
        
        # Bottle Neck
        convm = self.bottle_neck_block1(pool4, 8)
        convm = self.bottle_neck_block2(convm, 8)

        # Decoding Architecture
        de_conv4 = self.decoder_block(convm, 8, en_conv4)
        de_conv3 = self.decoder_block(de_conv4, 4, en_conv3)
        de_conv2 = self.decoder_block(de_conv3, 2, en_conv2)
        de_conv1 = self.decoder_block(de_conv2, 1, en_conv1)
        
        x = self.head(de_conv1)
        return x