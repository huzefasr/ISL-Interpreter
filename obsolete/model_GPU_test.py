from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from keras.utils import normalize
from keras.callbacks import TensorBoard,ModelCheckpoint
import h5py
import numpy as np
import tensorflow as tf
import pickle
import cv2
import time

# Load the dataset
conv_layer = [2,3,4]
dense_layer = [0,1,2]
conv_neuron = [+32,-32]
Dropout = [0.30,0.20]



X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))
modelname = "a-z_{}-25".format(int(time.time()))
tensorboard = TensorBoard(log_dir=f"log/{modelname}")
#board = TensorBoard(Log_dir="logs/{}".format(modelname))

# Scaling the data. /255 since data is image data
X = normalize(X, axis=1)
print("({})".format(X.shape[0]/2400)+str(X.shape))  # (2400,50,50,1) - n,y,x,c


model = Sequential()
for drop in Dropout:
    for layer in conv_layer:
        for dense in dense_layer:
            for neuron in conv_neuron:
                Name = f"drop-{drop}-layer_con{layer}-dense{dense}-neuron{neuron}.h5"
                '''
                for i in range(conv_layer-1):
                    model.add(Conv2D(64+neuron, (3, 3), input_shape=X.shape[1:]))  # 64 3,3
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.30)

                    model.add(Flatten())

                for i in range(dense_layer):
                    model.add(dense(32))
                    model.add(Activation('relu'))

                model.add(Dense(26))  # OG 1
                model.add(Activation('sigmoid'))

                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                model.fit(X, Y, batch_size=30, epochs=25, validation_split=0.2,callbacks=[tensorboard,check])  # OG 30
                model.save(Name)
                '''
                print(Name)

#model.add(Dense(128)) # 64 to 32

 # sigmoid


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# batch size should be kept a little low(20-200) to avoid negative results
check = ModelCheckpoint("keras-{epoch:02d}-{val_loss:.5f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit(X, Y, batch_size=30, epochs=25, validation_split=0.2,callbacks=[tensorboard,check])  # OG 30

'''
i = 20
while i <= 25:
    model.fit(X, Y, batch_size=30, epochs=i, validation_split=0.2, callbacks = [board])  # OG 30
    model.save("{}-{}).model".format(modelname,i))
    i = i+1
'''
