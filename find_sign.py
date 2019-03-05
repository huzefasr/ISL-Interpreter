import os
import sys
import numpy as np
import cv2


def files():
    path = os.getcwd()
    array = []

    path = os.path.join(path,"signdataset")
    os.chdir(path)
    files = os.listdir(path)
    for file in files:
        file = file.split('.')
        array.append(file[0])
        print(file)
    return array

#######################3
available_words = []
list_of_words = files()
sentence = input("Enter your sentence which you want to convert into signs: \n")
words = sentence.split(' ')

for word in words:
    if word in list_of_words:
        available_words.append(word)

new_path = os.path.join(os.getcwd(),"catavi")
print(new_path)
cap = cv2.VideoCapture(new_path)
key = cv2.waitKey()

while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if key == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
