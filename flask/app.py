import cv2
#import tensorflow as tf
#import keras
import h5py
import os
from methods import method_backproject
import pickle
import tensorflow as tf
import numpy as np
import cv2
import sys
from flask import Flask,redirect,render_template,Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
'''
@app.route('/add_gesture')
def add_gesture():
    return render_template('add_gesture.html')

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')
'''
@app.route('/recognize_gesture')
def recognize_gesture():
    return render_template('recognize_gesture.html')

@app.route('/capture_hist')
def capture_histogram():
    return render_template('capture_hist.html')

@app.route('/recognize_gest')
def recognize_gest():
    return render_template('recognize_gest.html')

@app.route('/call_to_press_c')
def call_press_c():
    #return press_c()
    capture_hist(c = 1)
    return render_template('recognize_gesture.html')

'''
def display(prediction,mask):
    blackboard = np.ones((150,150))
    previous = 0
    i = 1
    hit = 0
    char = ['-']+['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    font = cv2.FONT_HERSHEY_SIMPLEX
    for alphabet in prediction:
        if alphabet == 1:
            hit=i
        i=i+1
    cv2.putText(blackboard, f"{char[hit]}", (30,100), font, 5, (0, 255, 0),5)

    blackboard1 = cv2.resize(blackboard, (50,50))
    mask1 = cv2.resize(mask, (50,50))
    joint_image = np.hstack((blackboard1,mask1))

    imgencode2 = cv2.imencode('.jpg',joint_image)[1]
    stringData2 = imgencode2.tostring()
    yield (b'--frame\r\n' b'Content-type: text/plain\r\n\r\n'+stringData2+b'\r\n')
    return None

def prepare(mask):
    IMG_SIZE = 50
    #grayscale
    img_array = mask
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
'''

def prediction_method():
    roi_hist = pickle.load(open("hist.pickle",'rb'))
    model = tf.keras.models.load_model("a-z-17.model")
    cap = cv2.VideoCapture(0)
    i=0
    while True:
        _,frame = cap.read() # always remember to place inside while loop
        flip = cv2.flip(frame,1)
        y,x,c = flip.shape
        x1 = int(x/2)
        y1 = int(y/4)
        x2 = x1+300
        y2 = y1+200
        flip_crop = flip[y1:y2,x1:x2]
        rect = cv2.rectangle(flip, (x1,y1), (x2,y2), (255,0,0), 1)
        hsv_flip_crop = cv2.cvtColor(flip_crop,cv2.COLOR_BGR2HSV)
        #cv2.imshow('flip',flip_crop)
        mask = method_backproject(hsv_flip_crop,roi_hist)
        ###################
        IMG_SIZE = 50
        #grayscale
        new_array = cv2.resize(mask,(IMG_SIZE,IMG_SIZE))
        prep_mask =new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
        ###########################
        prediction = model.predict([prep_mask])

        #######################
        blackboard = np.zeros((150,150))
        previous = 0
        i = 1
        hit = 0
        char = ['-']+['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        font = cv2.FONT_HERSHEY_SIMPLEX
        for alphabet in prediction[0]:
            if alphabet == 1:
                hit=i
                previous = hit
            i=i+1
        print(char[hit])
        cv2.putText(blackboard, f"{char[hit]}", (30,100), font, 5, (255, 255, 255),5)
        #############################
        blackboard = cv2.resize(blackboard, (300,300))
        mask = cv2.resize(mask, (300,300))
        joint_image = np.hstack((blackboard,mask))
        imgencode2 = cv2.imencode('.jpg',joint_image)[1]
        stringData2 = imgencode2.tostring()
        yield (b'--frame\r\n' b'Content-type: text/plain\r\n\r\n'+stringData2+b'\r\n')

'''
        if i%3==0:
            i_copy = i
            IMG_SIZE = 50
            #grayscale
            img_array = mask
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            new_array = new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)
            #prediction = model.predict([prepare(mask)])
            prediction = model.predict([new_array])

            #display(prediction[0],mask)
            blackboard = np.ones((150,150))
            previous = 0
            i = 1
            hit = 0
            charprediction = np.array(prediction)
            prediction = prediction.astype(int)
            print(prediction)
            char= ['-']+['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
            hit_og = [prediction[0]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            for alphabet in prediction[0]:

                if alphabet == 1:
                    hit=i
                    previous = hit
                i=i+1
            if hit == 0:
                hit = previous
            cv2.putText(blackboard, f"{char[hit]}", (30,100), font, 5, (0, 255, 0),5)

            blackboard1 = cv2.resize(blackboard, (200,200))
            mask1 = cv2.resize(mask, (200,200))
            joint_image = np.hstack((mask1,blackboard1))

            imgencode2 = cv2.imencode('.jpg',joint_image)[1]
            stringData2 = imgencode2.tostring()
            yield (b'--frame\r\n' b'Content-type: text/plain\r\n\r\n'+stringData2+b'\r\n')

            i = i_copy

            prediction = np.array(prediction)
            prediction = prediction.astype(int)
            print(prediction)
            '''
#    cap.release()
#    cv2.destroyAllWindows()

def capture_hist(c,i=5):
    cap = cv2.VideoCapture(0)
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
        #cv2.imshow("RAW", flip)

        if i%5 == 0:
            roi = flip[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [
                                        180, 256], [0, 180, 0, 256])
            back = method_backproject(hsv_flip, roi_hist)
            pickle_out = open("hist.pickle", "wb")
            pickle.dump(roi_hist, pickle_out)
            print("histogram saved successfully")
            pickle_out.close()
        i = i+1

        imgencode = cv2.imencode('.jpg',flip)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n' b'Content-type: text/plain\r\n\r\n'+stringData+b'\r\n')

    return None
'''
def press_c():
    flip = capture_hist()
    y, x, c = flip.shape
    x1 = int(x / 2)
    y1 = int(y / 4)
    x2 = x1 + 70
    y2 = y1 + 200
    roi = flip[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [
                            180, 256], [0, 180, 0, 256])
    back = method_backproject(hsv_flip, roi_hist)
    return roi_hist

def press_s():
    roi_hist = press_c()
    pickle_out = open("hist.pickle", "wb")
    pickle.dump(roi_hist, pickle_out)
    print("histogram saved successfully")
    pickle_out.close()
    return None
'''

@app.route("/calc")
def calc():
    return Response( capture_hist(c = 0),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/calc2")
def calc2():
    return Response( prediction_method(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':

    app.run(debug=True )
