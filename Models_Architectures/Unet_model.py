import tensorflow as tf
from keras.layers import (InputLayer, Conv2D,BatchNormalization,
                          Dropout,MaxPooling2D,Conv2DTranspose,
                          concatenate)
from tensorflow.keras import Model
from mraunet_layers import UnetEncoderBlock, UnetBottleneck, UnetDecoderBlock 
from mramodule import MRAModule

# Define the RAUnet class which inherits from the Model class in Keras
class RAUnet(Model):
    def __init__(self):
        super(RAUnet, self).__init__()
        
        # Initialize the encoder blocks
        self.encoder_block1 = UnetEncoderBlock(64)
        self.endcoder_block2 = UnetEncoderBlock(128)
        self.endcoder_block3 = UnetEncoderBlock(256)
        self.endcoder_block4 = UnetEncoderBlock(512)
        
        # Initialize the decoder blocks
        self.decoder_block1 = UnetDecoderBlock(1024)
        self.decoder_block2 = UnetDecoderBlock(512)
        self.decoder_block3 = UnetDecoderBlock(256)
        self.decoder_block4 = UnetDecoderBlock(128)
        
        # Initialize the Multi-Resolution Attention (MRA) modules
        self.mra1 = MRAModule()
        self.mra2 = MRAModule()
        self.mra3 = MRAModule()
        self.mra4 = MRAModule()
        
        # Initialize the bottleneck layers
        self.bNeck1 = UnetBottleneck(1024)
        self.bNeck2 = UnetBottleneck(1024)
        
        # Initialize the final Convolutional layer
        self.header = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')
    
    
    # Overriding call function
    def call(self, inputs):
        
        # Encoding Architecture
        skipCon1, pool1 = self.encoder_block1()(inputs)
        skipCon2, pool2 = self.endcoder_block2()(pool1)
        skipCon3, pool3 = self.endcoder_block3()(pool2)
        skipCon4, pool4 = self.endcoder_block4()(pool3)

        # Skip Block
        skipCon4 = self.mra1()(skipCon1)
        skipCon2 = self.mra2()(skipCon2)
        skipCon3 = self.mra3()(skipCon3)
        skipCon4 = self.mra4()(skipCon4)
        
        # Bottle Neck
        bNeck1 = self.bNeck1()(pool4)
        bNeck2 = self.bNeck2()(bNeck1)

        # Decoding Architecture
        decOutput1 = self.UnetDecoderBlock()(bNeck2, skipCon1)
        decOutput2 = self.UnetDecoderBlock()(decOutput1, skipCon2)
        decOutput3 = self.UnetDecoderBlock()(decOutput2, skipCon3)
        decOutput4 = self.UnetDecoderBlock()(decOutput3, skipCon4)
        
        # Final output
        output = self.header(decOutput4)
        return output
