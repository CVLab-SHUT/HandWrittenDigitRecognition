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
    
if __name__ == "__main__":
    file_path = r"C:\Users\reza\Desktop\HandwrittenDigitRecognition\demo_codes\handwrittendigitrecognition\data_handling\data\Arabic.pkl"
    X_train, X_val, Y_train, Y_val, file_name = open_pickle(file_path)
    print(X_train, X_val, Y_train, Y_val, file_name)