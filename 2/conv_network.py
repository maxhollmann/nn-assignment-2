# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:36:07 2017

Convolutional Network Example

@author: Anonymous
"""

# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train_s = numpy.zeros((25000, 3, 32, 32), dtype = "uint8")
y_train_s = numpy.zeros((25000, 1), dtype = "uint8")
for i in numpy.unique(y_train):
    y_train_s[(i*2500):((i+1)*2500), 0] = numpy.array([i] * 2500)
    X_tmp = X_train[y_train[:, 0] == i, :, :, :]
    ind = numpy.random.permutation(numpy.arange(5000))
    X_tmp = X_tmp[ind[numpy.arange(2500)], : , :, :]
    X_train_s[(i*2500):((i+1)*2500), :, :, :] = X_tmp
    
    


# fix random seed for reproducibility
seed = 583
numpy.random.seed(seed)

# normalize inputs from 0-255 to 0.0-1.0
X_train_s = X_train_s.astype('float32')
X_test = X_test.astype('float32')
X_train_s = X_train_s / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train_s = np_utils.to_categorical(y_train_s)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 10
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# transform data to fit in structure
#X_train_s = X_train_s.transpose(0,3,1,2)
#X_test = X_test.transpose(0,3,1,2)

# Fit the model
model.fit(X_train_s, y_train_s, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

