from PIL import Image
import numpy as np
import pickle
import bz2
import os

directory = r"C:\Users\reza\Desktop\Handwritten digit recognition\demo_codes\new repository\Datasets"

data_path = {
    "Arabic": directory + r"\Arabic.pkl",
    "Chinese": directory + r"\Chinese.pkl",
    "English": directory + r"\English.pkl", 
    "Gujarati": directory + r"\Gujarati.pkl",
    "Gurmukhi": directory + r"\Gurmukhi.npz",
    "ISI_Bangla": directory + r"\ISI_Bangla.npz",
    "Kannada": directory + r"\Kannada.npz",
    "Tibetan": directory + r"\Tibetan.npz",
    "Urdu": directory + r"\Urdu.npz",
    "ARDIS": directory + r"\ARDIS.npz",
    "BanglaLekha": directory + r"\BanglaLekha.npz"
}


def load_data(path):
    if os.path.splitext(path)[1] == ".npz":
        data = np.load(path)
    elif os.path.splitext(path)[1] == ".pkl":
        with bz2.BZ2File(path, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
            data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
    else:
        raise ValueError("Invalid file extension")
    return data


def save_data(data, target_path, n_class = 10):
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    train_path = os.path.join(target_path, "train")
    test_path = os.path.join(target_path, "test")
    
    # make test and train folder
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        
    # make folders for classes and save images 
    for i in range(n_class):
        print(i)
        new_class_dir = os.path.join(train_path, str(i))
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)
        
    for i in range(len(X_train)):
        img = Image.fromarray(X_train[i])
        filename = os.path.join(train_path, str(y_train[i]), str(i) + ".png")
        img.save(filename)

    # make folders for classes and save images 
    for i in range(n_class):
        new_class_dir = os.path.join(test_path, str(i))
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)
        
    for i in range(len(X_test)):
        img = Image.fromarray(X_test[i])
        filename = os.path.join(test_path, str(y_test[i]), str(i) + ".png")
        img.save(filename)
        
data = load_data(data_path["Chinese"])
save_data(data, directory)