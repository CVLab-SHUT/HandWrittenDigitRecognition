from scipy.ndimage import rotate
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import cv2


def upsample(X_train, size):
    return np.array([cv2.resize(img, (size, size)) for img in X_train])

def augmentation(input_degrees, X_train, y_train, size):
    # Calculate the total number of augmented images
    total_images = X_train.shape[0] * len(input_degrees)

    # Create an empty numpy array to store the augmented images
    augmented_images = np.zeros((total_images, size, size))

    # Generate augmented images for each input image
    for i, (image, angle) in enumerate(zip(np.repeat(X_train, len(input_degrees), axis=0), np.tile(input_degrees, X_train.shape[0]))):
        augmented_images[i] = rotate(image, angle, reshape=False)

    # Create the augmented labels array
    augmented_labels = np.repeat(y_train, len(input_degrees))

    # Verify the shape of the augmented_images and augmented_labels arrays
    print(augmented_images.shape)
    print(augmented_labels.shape)

    return augmented_images, augmented_labels
    
    
def plot_images_labels(images, labels):
    _, axes = plt.subplots(nrows=3, ncols=4, figsize=(5, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()
    
    
def my_callback(filepath):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        mode='max',
        save_weights_only=False
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        mode='max'
    )

    return [checkpoint]


def evaluate_model(X_test, Y_test, model):
    score=model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def confusion_matrix(Y_hat,Y_test):
    # Convert the multilabel format of Y_hat to a one-dimensional array of labels
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_hat_labels = np.argmax(Y_hat, axis=1)
    cm = confusion_matrix(Y_test_labels, Y_hat_labels)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()