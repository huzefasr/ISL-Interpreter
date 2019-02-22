import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from methods import *
import pickle
import time

# Tried but obsolete so has been commented out


def draw_rect(img, mode):
    y, x, c = img.shape
    if mode == 1:
        d = 5
        x = int(x / 2) + 30
        y = int(y / 2)
        m = x
        imgCrop = None
        imgCrop2 = None
        for i in range(2):
            for j in range(2):
                if np.any(imgCrop == None):
                    imgCrop = img[y:y + 10, x:x + 10]
                    imgCrop = np.hstack((imgCrop, img[y:y + 10, x:x + 10]))
                else:
                    imgCrop = np.hstack((imgCrop, img[y:y + 10, x:x + 10]))
                    imgCrop = np.hstack((imgCrop, img[y:y + 10, x:x + 10]))
                cv2.rectangle(img, (x, y), (x + 10, y + 10), (255, 0, 0), 1)
                x = x + 10 + d
                if j == 1:
                    imgCrop = np.hstack((imgCrop, img[y:y + 10, x:x + 10]))
                    cv2.rectangle(img, (x, y), (x + 10, y + 10),
                                  (255, 0, 0), 1)
            if np.any(imgCrop2 == None):
                imgCrop2 = imgCrop
            else:
                imgCrop2 = np.vstack((imgCrop2, imgCrop))
            imgCrop = None
            x = m
            y = y + 10 + dx
            return(imgCrop2, img)
        else:
            x1 = int(x / 2)
            y1 = int(y / 4)
            x2 = x1 + 70
            y2 = y1 + 200
            rect = cv2.rectangle(flip, (x1, y1), (x2, y2), (255, 0, 0), 1)
            return img


def nothing(x):
    pass


def read_image():
    cap = cv2.VideoCapture(0)
    keyc, keys = False, False
    while True:
        key = cv2.waitKey(1)
        _, frame = cap.read()
        flip = cv2.flip(frame, 1)
        hsv_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)

        #flip = draw_rect(flip, mode = 0)
        y, x, c = flip.shape
        x1 = int(x / 2)
        y1 = int(y / 4)
        x2 = x1 + 70
        y2 = y1 + 200
        rect = cv2.rectangle(flip, (x1, y1), (x2, y2), (255, 0, 0), 1)
        #roi,flip = draw_rect(flip)

        cv2.imshow("RAW", flip)
        '''
		Here we check for input key
		'''
        if key == ord('l'):
            roi_hist = pickle.load(open("hist.pickle", "rb"))
            cap.release()
            cv2.destroyAllWindows()
            path_og = path()
            save_images(path_og, roi_hist)
        if key == ord('c'):
            keyc = True
            roi = flip[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [
                                    180, 256], [0, 180, 0, 256])

        if key == ord('s'):
            pickle_out = open("hist.pickle", "wb")
            pickle.dump(roi_hist, pickle_out)
            pickle_out.close()
            cap.release()
            cv2.destroyAllWindows()
            ########
            path_og = path()
            save_images(path_og, roi_hist)

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

def path():
    path = os.getcwd()
    path = os.path.join(path, 'dataset')
    word = input("Enter the letter you want to save:\n").lower()
    if os.path.exists(path):
        path = os.path.join(path, word)
        if not(os.path.exists(path)):
            os.mkdir(path)
    else:
        os.mkdir(path)
        path = os.path.join(path, word)
        os.mkdir(path)

    return path


def save_images(path_og, roi_hist):
    cap = cv2.VideoCapture(0)
    keyc = False
    keyx = False
    new_file = []
    frame_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _, frame = cap.read()
        key = cv2.waitKey(1)
        flip = cv2.flip(frame, 1)
        y, x, c = flip.shape
        x1 = int(x / 2)
        y1 = int(y / 4)
        x2 = x1 + 300
        y2 = y1 + 200
        hsv_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)
        flip = method_backproject(hsv_flip, roi_hist)
        rect = cv2.rectangle(flip, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imshow("flip", flip)
        if key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            path_og = path()
            save_images(path_og, roi_hist)
        if key == ord('c'):
            keyc = True
        if keyc:
            frame_count = frame_count + 1
            if frame_count > 70:

                files = os.listdir(path_og)
                ext = ".png"
                if files == []:
                    name = 0
                else:
                    for file in files:
                        file = file.split('.')
                        new_file.append(int(file[0]))
                    new_file.sort()
                    name = new_file[-1]
                for n in range(300):
                    _, frame = cap.read()
                    flip = cv2.flip(frame, 1)
                    hsv_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)
                    flip = method_backproject(hsv_flip, roi_hist)
                    flip = flip[y1:y2, x1:x2]

                    name = name + 1
                    imgname = str(name) + '.png'
                    imgname = os.path.join(path_og, imgname)
                    flip = cv2.resize(flip, (50, 50))
                    cv2.imwrite(imgname, flip)

                    normal = cv2.flip(flip,1)
                    name = name + 1
                    imgname = str(name) + '.png'
                    imgname = os.path.join(path_og, imgname)
                    cv2.imwrite(imgname, normal)
                    print(imgname)
                keyc = False
        if key == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_image()
