import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from handle_data import save_image_digit, create_directory_digit
import numpy as np

class Filter:
    def __init__(self, directory, target_size=(128, 128), batch_size=1):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255, 
                                          rotation_range=25,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          fill_mode='nearest')

    def flow_from_directory(self):
        for class_dir in os.listdir(self.directory):
            class_path = os.path.join(self.directory, class_dir)
            num_samples = len(os.listdir(class_path))
            # Use data augmentation
            generator = self.datagen.flow_from_directory(
                self.directory,
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='categorical'
            )
                
        return generator
    
    
    def filter_10000(generator):
        random_samples = []
        for _ in range(10000):
            # Retrieve a batch of data from the generator
            batch_data, _ = generator.next()
            # Append the batch data to the list
            random_samples.extend(batch_data)
        return np.array(random_samples)


if __name__ == "__main__":
    create_directory_digit("dest_path", "ISI_Bangla")
    filter = Filter("path")
    generator = filter.flow_from_directory()
    
    