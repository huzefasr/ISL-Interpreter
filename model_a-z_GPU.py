from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from keras.utils import normalize
from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
import pickle
import cv2
import time
from keras.callbacks import TensorBoard
# Load the dataset

X = pickle.load(open("X_ab.pickle", "rb"))
Y = pickle.load(open("Y_ab.pickle", "rb"))
modelname = "a-z_GPU{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"log/{modelname}")
#board = TensorBoard(Log_dir="logs/{}".format(modelname))

# Scaling the data. /255 since data is image data
X = normalize(X, axis=1)
print("({})".format(X.shape[0]/2400)+str(X.shape))  # (2400,50,50,1) - n,y,x,c


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))  # 64 3,3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #
model.add(Dropout(0.40))
model.add(Conv2D(128, (3, 3)))  # 5,5
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# EXTRA
model.add(Conv2D(64, (3, 3)))  # 5,5
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



# remember to  flatten the data since the data is 2d and dense accepts 1d data
model.add(Flatten())
#model.add(Dense(128)) # 64 to 32

model.add(Dense(26))  # OG 1
model.add(Activation('sigmoid'))  # sigmoid


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# batch size should be kept a little low(20-200) to avoid negative results

model.fit(X, Y, batch_size=30, epochs=20, validation_split=0.2,callbacks=[tensorboard])  # OG 30
model.save(f"{modelname}-20")

'''
i = 8
while i <= 10:
    model.fit(X, Y, batch_size=30, epochs=i, validation_split=0.2, callbacks = [board])  # OG 30
    model.save("{}-{}).model".format(modelname,i))
    i = i+1
'''
