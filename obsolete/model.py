from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import normalize
from keras import regularizers
import numpy as np
import tensorflow as tf
import pickle
import cv2
from create_training_data import category_data
_,categories = category_data()
# Load the dataset

X = pickle.load(open("X_ab.pickle", "rb"))
Y = pickle.load(open("Y_ab.pickle", "rb"))

# Scaling the data. /255 since data is image data
X = normalize(X, axis=1)  # (2400,50,50,1) - n,y,x,c


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]),)  # 64 3,3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #

#model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3),kernel_regularizer=regularizers.l2(0.01)))  # 5,5 128
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25)) #0.25

# EXTRA
model.add(Conv2D(64, (3, 3)))  # 5,5
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



# remember to  flatten the data since the data is 2d and dense accepts 1d data
model.add(Flatten())
#model.add(Dense(16)) # 64 to 32

model.add(Dense(categories))  # OG 1
model.add(Activation('sigmoid'))  # sigmoid


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# batch size should be kept a little low(20-200) to avoid negative results
model.fit(X, Y, batch_size=30, epochs=1, validation_split=0.1,verbose=1)  # OG 30

model.save("model_test_ab.model")
