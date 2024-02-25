import tensorflow as tf
# The call function is implemented leveraging polymorphysm
class ClassifierBase(tf.keras.Model):
    def __init__(self, num_classes):
        super(ClassifierBase, self).__init__()
        self.num_classes = num_classes

    def call(self, inputs):
        raise NotImplementedError
    
    def compile_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train_model(self, train_data, epochs, validation_data):
        self.model.fit(train_data, epochs=epochs, validation_data=validation_data)
