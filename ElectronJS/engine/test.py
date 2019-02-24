import cv2
import numpy as np
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    cam = cv2.VideoCapture(0)
    key = cv2.waitKey(0)
    while True:
        _,frame = cam.read()
        cv2.imshow("frame",frame)
        if key == ord('c'):
            break
    cv2.destroyAllWindows()
    return("this is a  webpage")


if __name__=="__main__":
    app.run(debug=True)
