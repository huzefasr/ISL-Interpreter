from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from keras.utils import normalize
from keras.callbacks import TensorBoard,ModelCheckpoint
import h5py
import os
#import keras
#import tensorflowjs as tfjs
import numpy as np
#import tensorflow as tf
import pickle
import cv2
import time

# Load the dataset

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

# Load the outputs

path = os.path.join(os.getcwd(),"dataset")
files = len(os.listdir())

# Setting Parameters:

Dropout_p = 0.30
output = files
model_name = "test.h5"


# Scaling the data. /255 since data is image data
X = normalize(X, axis=1)
print("({})".format(X.shape[0]/2400)+str(X.shape))  # (2400,50,50,1) - n,y,x,c


model = Sequential()

# Layer 1 Conv2D
model.add(Conv2D(64, (5, 5), input_shape=X.shape[1:]))  # 64 3,3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout layer
model.add(Dropout(Dropout_p))

# Layer 2 Conv2D
model.add(Conv2D(128, (5, 5)))  # 5,5
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout Layer
model.add(Dropout(Dropout_p))

# Layer 3 Conv2D
model.add(Conv2D(64, (5, 5)))  # 5,5
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Converting to 1D array
model.add(Flatten())

model.add(Dense(output))  # OG 1
model.add(Activation('sigmoid'))  # sigmoid


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.save("test.h5")

# Tensorboard and Checkpoint Callbacks
#tensorboard = TensorBoard(log_dir=f"log/{model_name}")
#board = TensorBoard(Log_dir="logs/{}".format(modelname))
#check = ModelCheckpoint("a-z-newl-{epoch:02d}-{val_loss:.5f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.fit(X, Y, batch_size=30, epochs=1, validation_split=0.2)  #,callbacks=[tensorboard,check]
