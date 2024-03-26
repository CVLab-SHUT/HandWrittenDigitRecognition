import tensorflow as tf
from keras.layers import (InputLayer, Conv2D, BatchNormalization,
                          Dropout, MaxPooling2D, Conv2DTranspose,
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
        self.encoder_block2 = UnetEncoderBlock(128)
        self.encoder_block3 = UnetEncoderBlock(256)
        self.encoder_block4 = UnetEncoderBlock(512)
        
        # Initialize the decoder blocks
        self.decoder_block1 = UnetDecoderBlock(512)
        self.decoder_block2 = UnetDecoderBlock(256)
        self.decoder_block3 = UnetDecoderBlock(128)
        self.decoder_block4 = UnetDecoderBlock(64)
        
        # Initialize the Multi-Resolution Attention (MRA) modules
        self.mra_module1 = MRAModule()
        self.mra_module2 = MRAModule()
        self.mra_module3 = MRAModule()
        self.mra_module4 = MRAModule()
        
        # Initialize the bottleneck layers
        self.bottleneck1 = UnetBottleneck(1024)
        self.bottleneck2 = UnetBottleneck(1024)
        
        # Initialize the final Convolutional layer
        self.final_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')
    
    
    # Overriding call function
    def call(self, inputs):
        
        # Encoding Architecture
        skip_connection1, pooled_output1 = self.encoder_block1(inputs)
        skip_connection2, pooled_output2 = self.encoder_block2(pooled_output1)
        skip_connection3, pooled_output3 = self.encoder_block3(pooled_output2)
        skip_connection4, pooled_output4 = self.encoder_block4(pooled_output3)

        # Skip Block
        skip_connection1 = self.mra_module1(skip_connection1)
        skip_connection2 = self.mra_module2(skip_connection2)
        skip_connection3 = self.mra_module3(skip_connection3)
        skip_connection4 = self.mra_module4(skip_connection4)
        
        # Bottle Neck
        bottleneck_output1 = self.bottleneck1(pooled_output4)
        bottleneck_output2 = self.bottleneck2(bottleneck_output1)

        # Decoding Architecture
        decoder_output1 = self.decoder_block1([bottleneck_output2, skip_connection4])
        decoder_output2 = self.decoder_block2([decoder_output1, skip_connection3])
        decoder_output3 = self.decoder_block3([decoder_output2, skip_connection2])
        decoder_output4 = self.decoder_block4([decoder_output3, skip_connection1])
        
        # Final output
        final_output = self.final_layer(decoder_output4)
        return final_output
