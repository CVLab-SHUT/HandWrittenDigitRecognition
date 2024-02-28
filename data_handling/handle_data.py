import os
import numpy as np
import cv2
import bz2
import pickle

def open_pickle(file_path):
    with bz2.BZ2File(file_path, "rb") as f2:
        return pickle.load(f2)
        
def open_numpy(file_path):
    data = np.load(file_path, allow_pickle =  True)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def read_data(path):
    files_list = os.listdir(path)
    languages = []
    for file in files_list:
        
        if file == "Gurmukhi.npy":
            data = np.load(os.path.join(path, file), allow_pickle = True)
            X_train, y_train, X_test, y_test = data
        elif file.split(".")[1] == "pkl":
            X_train, y_train, X_test, y_test = open_pickle(os.path.join(path, file))
        else:
            X_train, y_train, X_test, y_test = open_numpy(os.path.join(path, file))
            
        languages.append(X_train, y_train, X_test, y_test)
    # yield X_train, y_train, X_test, y_test
    return languages

def create_directory(file_list, path):
    for file in file_list:
        os.mkdir(os.path.join(path, file.split(".")[0]))
        print(os.listdir(path))
        
if __name__ == "__main__":
    languages_data = read_data(r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data")
    saved_path = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling"
    create_directory(saved_path, read_data(r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data"))
    # file_path = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data\BanglaLekha.npz"
    # data = np.load(file_path, allow_pickle =  True)
    # print(data)
    # print(data['X_train'], data['y_train'], data['X_test'], data['y_test'])