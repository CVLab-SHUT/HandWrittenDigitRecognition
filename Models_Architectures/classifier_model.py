import tensorflow as tf
from keras.layers import (InputLayer, Conv2D,BatchNormalization,
                          Dropout,MaxPooling2D, Dense, Flatten)
from module import Module


# The call function is implemented leveraging polymorphysm
class ClassifierBase(tf.keras.Model):
    def __init__(self, n_classes):
        super(ClassifierBase, self).__init__()
        self.num_classes = n_classes

    def call(self, inputs):
        raise NotImplementedError
    
    def compile_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train_model(self, train_data, epochs, validation_data):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data)

    def evaluate_model(self, test_data):
        test_loss, test_acc = self.model.evaluate(test_data)
        print('Test accuracy:', test_acc)


class LanguageClassifier(ClassifierBase):
    def __init__(self, i_shape, n_neurons = 64, coef = 1, num_classes = 12):
        super(LanguageClassifier, self).__init__()
        self.input = InputLayer(input_shape = i_shape)
        self.n_neurons = n_neurons
        self.coef = coef
        self.block1 = self.conv_block()
        self.block2 = self.conv_block()
        self.block3 = self.conv_block()
        self.header = self.head()
        
        
    def conv_block(self, inputs):
        conv = Conv2D(self.n_neurons , (3, 3), activation='relu', padding='same')(inputs)
        conv = BatchNormalization()(conv)
        conv = Conv2D(self.n_neurons , (3, 3), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPooling2D((2, 2))(conv)
        conv = Module(conv, self.n_neurons, self.coef).call()
        return conv
        
        
    def head(self, inputs):
        flatten = Flatten()(inputs)
        dense = Dense(512, activation = 'relu')(flatten)
        dense = Dropout(0.25)(dense)
        dense = Dense(256, activation = 'relu')(dense)
        dense = Dropout(0.25)(dense)
        dense = Dense(128, activation = 'relu')(dense)
        dense = Dropout(0.25)(dense)
        dense = Dense(10, activation = 'softmax')(dense)
        return dense


    # overriding call function
    def call(self):
        x = self.block1(self.input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x
    
        
if __name__ == "__main__":
    classifier = LanguageClassifier((128,128,1), n_neurons = 64, coef = 1, num_classes = 12)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # classifier.fit(train_data, epochs=10, validation_data=validation_data)
