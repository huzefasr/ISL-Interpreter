from flask import Flask,render_template,request
from scipy.misc import imsave,imread,imresize
import numpy as np
import keras.models
import os
import sys
import re
sys.path.append(os.path.abspath('./model'))
from load import *

app = Flask(__name__)

global model, graph
model, graph = init()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ["GET","POST"])
def predict():
    imgData = request.get_data()

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host=(0.0.0.0), port = port)
