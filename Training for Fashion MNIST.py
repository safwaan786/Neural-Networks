from Neural_Networks import * 
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import numpy as np
import matplotlib.pyplot as plt 

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip' 
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print('Done!')


def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X, y = [], []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test


X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Preprocessing data to the range (-1,1) and then flattening
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Data Shuffling
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(decay=1e-3), accuracy=Accuracy_Categorical())

model.finalise()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

parameters = model.get_parameters()

#model.save_parameters('fashion_mnist.parms')

#model.load_parameters('fashion_mnist.parms')

model.save('fashion_mnist.model')





