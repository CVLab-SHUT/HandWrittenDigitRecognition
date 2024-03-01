import os
import cv2
import bz2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# to load a single pickle file
def open_pickle(file_path):
    with bz2.BZ2File(file_path, "rb") as f2:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        return pickle.load(f2), file_name


# to load a single numpy file        
def open_numpy(file_path):
    data = np.load(file_path, allow_pickle=True)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return data['X_train'], data['y_train'], data['X_test'], data['y_test'], file_name


# to load all dataset in a directory
def read_data(path, files_list):
    languages = []
    for file in files_list:
        file_path = os.path.join(path, file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension == ".pkl":
            try:
                X_train, y_train, X_test, y_test, file_name = open_pickle(file_path)
            except:
                try:
                    print("test here is printed")
                    data, file_name = open_pickle(file_path)
                    X_train, y_train, X_test, y_test = data
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
        else:
            try:
                X_train, y_train, X_test, y_test, file_name = open_numpy(file_path)
            except Exception as e:
                    print(f"Error processing file {file}: {e}")
        languages.append((file_name, X_train, y_train, X_test, y_test))
        print(len(languages))
    return languages


# to create a directory for digit classification
def create_directory_digit(path, file_name):
    folder_path = os.path.join(path, file_name)
    train_dir_path = os.path.join(folder_path, 'training_dir')
    validation_dir_path = os.path.join(folder_path, 'validation_dir')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        os.mkdir(train_dir_path)
        os.mkdir(validation_dir_path)
        for i in range(10):
            os.mkdir(os.path.join(train_dir_path, str(i)))
            os.mkdir(os.path.join(validation_dir_path, str(i)))
    return train_dir_path, validation_dir_path
    
# save images in directory for digit classification
def save_image_digit(X_train, y_train, path, file_name):
    train_dir_path, validation_dir_path = create_directory_digit(path, file_name)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.2,
                                                      random_state=42)
    for i, (img, label) in enumerate(zip(X_train, y_train)):
        img_path = os.path.join(path, file_name, train_dir_path , str(label), f"img{i + 1}.jpg")
        cv2.imwrite(img_path, img)
        
    for i, (img, label) in enumerate(zip(X_val, y_val)):
        img_path = os.path.join(path, file_name, validation_dir_path , str(label), f"img{i + 1}.jpg")
        cv2.imwrite(img_path, img)
        

# to create a directory for language classification
def create_directory_language(path):
    folder_path = os.path.join(path, "language_dataset")
    train_dir_path = os.path.join(folder_path, 'training_dir')
    validation_dir_path = os.path.join(folder_path, 'validation_dir')
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        os.mkdir(train_dir_path)
        os.mkdir(validation_dir_path)
        for lan in ["Arabic","BanglaLekha","English","Gurmukhi",
                    "Kannada","Tibetan","ARDIS", "Chinese",
                    "Gujarati","ISI_Bangla","Persian","Urdu"]:
            
            os.mkdir(os.path.join(train_dir_path, lan))
            os.mkdir(os.path.join(validation_dir_path, lan))
            
    return train_dir_path, validation_dir_path
    
    
# save images in directory for language classification
def save_image_language(path):
    files_list = os.listdir(path)
    train_dir_path, validation_dir_path = create_directory_language(path)
    languages = read_data(path, files_list)
    
    for lan in languages:
        data = lan[1]
        file_name = lan[0]
        X_train, X_val = train_test_split(data,
                                        test_size=0.2,
                                        random_state=42)
        
        for i, img in enumerate(X_train):
            img_path = os.path.join(train_dir_path, file_name, f"img{i + 1}.jpg")
            plt.imsave(img_path, img, cmap='gray')
            
        for i, img in enumerate(X_val):
            img_path = os.path.join(validation_dir_path, file_name, f"img{i + 1}.jpg")
            plt.imsave(img_path, img, cmap='gray')

# arabic, english, gujarati, gurmukhi
        
        
if __name__ == "__main__":
    # languages_data = read_data(r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data")
    # saved_path = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling"
    # create_directory(saved_path, read_data(r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data"))
    file_path = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data\BanglaLekha.npz"
    file_path_save = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data"
    # data = np.load(path + "\Gurmukhi.npz", allow_pickle=True)
    # print(data.files)
    # X_train, y_train, X_test, y_test, file_name = open_numpy(file_path)
    # save_image_digit(X_train, y_train, file_path_save, file_name)
    save_image_language(file_path_save)
    # create_directory_language(file_path_save)
    # print(X_train.shape)
        