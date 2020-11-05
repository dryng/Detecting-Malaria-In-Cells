import os
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize


# data processing
def load_data():
    X = []
    y = []
    data_dir = "/Users/danny/Courses/Udemy/TF_2/cell_images"
    test_path = data_dir + '/test/'
    train_path = data_dir + '/train/'
    # add test parasitized
    shape = (130,130,3)
    for filename in os.listdir(test_path + 'parasitized'):
        if '.png' in test_path + 'parasitized/' + filename or '.jpeg' in test_path + 'parasitized/' + filename:
            img = imread(test_path + 'parasitized/' + filename)
            img = resize(img, shape)
            X.append(img)
            y.append(1)

    # add test uninfected
    for filename in os.listdir(test_path + 'uninfected'):
        if '.png' in test_path + 'uninfected/' + filename or '.jpeg' in test_path + 'uninfected/' + filename:
            img = imread(test_path + 'uninfected/' + filename)
            img = resize(img, shape)
            X.append(img)
            y.append(0)

    # add train parasitized
    for filename in os.listdir(train_path + 'parasitized'):
        if '.png' in train_path + 'parasitized/' + filename or '.jpeg' in train_path + 'parasitized/' + filename:
            img = imread(train_path + 'parasitized/' + filename)
            img = resize(img, shape)
            X.append(img)
            y.append(1)

    # add train uninfected
    for filename in os.listdir(train_path + 'uninfected'):
        if '.png' in train_path + 'uninfected/' + filename or '.jpeg' in train_path + 'uninfected/' + filename:
            img = imread(train_path + 'uninfected/' + filename)
            img = resize(img, shape)
            X.append(img)
            y.append(0)

    #X, y = map(list, zip(*X))

    X = np.array(X)
    y = np.array(y)
    
    # make dataset smaller
    X = np.array_split(X, 5)[0]
    y = np.array_split(y, 5)[0]
   

    classes = ['uninfected', 'parasitized']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return X_train, X_test, y_train, y_test, classes