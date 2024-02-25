import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from functions import upsample

img_size = 128
language = "persian"

rotation_range = {
    "Arabic":[-35,-30,-25,-20 ,-15,-10,-5,
              5, 10, 15, 20, 25, 30, 35],
    "Aedis":[-30,-15, 15, 30],
    "BanglaLekha":[-30,30],
    "English":[-30,30],
    "Chinese":[30],
    "Gurmukhi":[-35, -30, -25, -20,-15,-10,
                -5, 5, 10, 15, 20, 25, 30, 35],
    "Persian":[-30, 30],
    "ISI_Bangla":[-30, 30],
    "Kannada":[30],
    "Tibetan":[-30, 30],
    "Urdu":[-30,-20, -10, 15,30],
    "Gujarati":[-30,-20 ,-10, 10, 20,30]
}

# Generate batches of augmented images
train_gen = ImageDataGenerator(
    rotation_range = rotation_range[language],
    validation_split=0.2,
    rescale = 1./255
)

# Generate batches of augmented images
test_gen = ImageDataGenerator(
    rescale = 1./255
)

train_generator = train_gen.flow_from_directory('train', target_size = img_size, batch_size = 32, class_mode = 'categorical')
test_generator = test_gen.gen.flow_from_directory('test',target_size = img_size, batch_size = 32, class_mode = 'categorical')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=val_data,
        validation_steps=800,
        verbose = 1
)
