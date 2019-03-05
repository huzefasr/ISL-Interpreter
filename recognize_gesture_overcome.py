import cv2
import tensorflow as tf
import keras
import h5py
import os
from methods import method_backproject
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyttsx3
engine = pyttsx3.init()
path = os.getcwd()
path = os.path.join(path,'dataset')
category = sorted(os.listdir(path))
print(category)
def say(text):
    engine.say(text)
    engine.runAndWait()

def display(prediction):
    blackboard = np.ones((150,150))
    previous = 0
    i = 1
    hit = 0
    char = ['-']+category
    hit_og = [prediction]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for alphabet in prediction:
        if alphabet == 1:
            hit=i
            previous = hit
        i=i+1
    if hit == 0:
        hit = previous
    letter = char[hit]
    cv2.putText(blackboard, f"{letter}", (30,100), font, 5, (0, 255, 0),5)

    ### SPEACH
    cv2.imshow("Gesture",blackboard)
    return letter



def display(prediction):
    blackboard = np.ones((150,150))
    previous = 0
    i = 1
    hit = 0
    char = ['-']+category
    hit_og = [prediction]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for alphabet in prediction:
        if alphabet == 1:
            hit=i
            previous = hit
        i=i+1
    if hit == 0:
        hit = previous
    letter = char[hit]
    cv2.putText(blackboard, f"{letter}", (30,100), font, 5, (0, 255, 0),5)

    ### SPEACH
    cv2.imshow("Gesture",blackboard)
    return letter


def prepare(mask):
    IMG_SIZE = 50
    #grayscale
    img_array = mask
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

def prediction_method():
    previous = 0
    j = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    roi_hist = pickle.load(open("hist.pickle",'rb'))
    model = keras.models.load_model("a-z.h5")
    cap = cv2.VideoCapture(0)
    i = 1
    hit = 0
    keyw = False
    j = 0
    width = 30
    word = np.ones((200,600))
    array = []
    printed_letter = '-'
    letter = '-'
    space = False
    word_string = ''
    keys = False
    while True:
        _,frame = cap.read()
        key = cv2.waitKey(1) # always remember to place inside while loop
        flip = cv2.flip(frame,1)
        y,x,c = flip.shape
        x1 = int(x/2)
        y1 = int(y/4)
        x2 = x1+300
        y2 = y1+200
        count = 0
        flip_crop = flip[y1:y2,x1:x2]
        rect = cv2.rectangle(flip, (x1,y1), (x2,y2), (255,0,0), 1)
        hsv_flip_crop = cv2.cvtColor(flip_crop,cv2.COLOR_BGR2HSV)
        cv2.imshow('flip',flip)
        mask = method_backproject(hsv_flip_crop,roi_hist)
        cv2.imshow('mask',mask)
        prediction = model.predict([prepare(mask)])
        prediction = prediction[0]

        old_letter = array.append(letter)
        if len(array) == 35:
            array = array[1:]
        print(array)
        letter = display(prediction)
        print(letter)

        for old_letter in array:
            if letter == old_letter:
                count = count + 1
        print(count)
        if key == ord('w'):
            keyw = True
        if key == ord(' '):
            space = True
        if key == ord('s'):
            say(word_string)
        if keyw:
            if letter != "-" and count > 30:
                say(letter)
                if space:
                    word_string = word_string + " "
                else:
                    word_string = word_string + letter
                if keys:
                    print("s is called")

                print("this is the string created"+str(word_string))
                if space:
                    width = width + 70
                    space = False
                cv2.putText(word, f"{letter}", (width,100), font, 2, (0, 255, 0),2)

                array = []
                width = width + 30
            cv2.imshow("word",word)
        if key == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()

def capture_hist():
    cap = cv2.VideoCapture(0)
    keyc, keys = False, False
    while True:
        key = cv2.waitKey(1)
        _, frame = cap.read()
        flip = cv2.flip(frame, 1)
        hsv_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)
        y, x, c = flip.shape
        x1 = int(x / 2)
        y1 = int(y / 4)
        x2 = x1 + 70
        y2 = y1 + 200
        rect = cv2.rectangle(flip, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imshow("RAW", flip)
        if key == ord('c'):
            keyc = True
            roi = flip[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [
                                    180, 256], [0, 180, 0, 256])
        if key == ord('s'):
            pickle_out = open("hist.pickle", "wb")
            pickle.dump(roi_hist, pickle_out)
            print("histogram saved successfully")
            pickle_out.close()
            cap.release()
            cv2.destroyAllWindows()
            break
        if key == ord('x'):
            cap.release()
            cv2.destroyAllWindows()
            break


        '''
		depending on key pressed we take the action we want to
		'''
        if keyc:
            back = method_backproject(hsv_flip, roi_hist)
            cv2.imshow("back project", back)

####CODDEEEEE
ch = input("Do you wish to create histogram or use existing\n(y/n)")
if ch == 'y':
    capture_hist()

prediction_method()
