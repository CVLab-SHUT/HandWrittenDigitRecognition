import os
import random
import numpy as np
from sklearn.utils import shuffle

# Set the path to the source folder
source_folder = r'C:\Reza\FunSpace\Digital_Generated_Images\numbers'

# Get a list of all the image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

image_files = np.array(image_files)

# Generate a random number for the file saving names
random_number = random.randint(1000, 9999)

for i in range(100000):
    # Shuffle the image files in a random order
    np.random.shuffle(image_files)
    
    # Set the path to the destination folder
    destination_folder = r'C:\Reza\FunSpace\Digital_Generated_Images\suffeled_numbers'
    
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Move the shuffled image files to the destination folder
    for file in image_files:
        os.rename(os.path.join(source_folder, file), os.path.join(destination_folder, str(random_number)+file))