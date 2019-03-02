import os
import cv2
import matplotlib.pyplot as plt
from colorama import Fore
import random
import numpy as np
import pickle
import time
start = time.time()
def category_data():
    path = os.getcwd()
    path_dataset = os.path.join(path,'dataset')
    char_folders = os.listdir(path_dataset)

    #categories = ['a','b','c','d']
    categories = sorted(char_folders)
    #categories = categories[:15]
    print(categories)
    return path_dataset,categories
training_data = []

def create_training_data():
    path_dataset,categories = category_data()

    for category in categories:
        path_categories = os.path.join(path_dataset,category)
        index = categories.index(category)
        for img in os.listdir(path_categories):
            try:
                path = os.path.join(path_categories,img)
                if path.endswith('.png'):
                    img_array = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    training_data.append([img_array,index])
            except Exception as e:
                print(Fore.RED+path)
        print(Fore.GREEN+f"imported all data from cateogry {category}")



#For every object in the training dataset
#training_data[element][data type*]
#0 - Stores the values of all images
#1 -  stores the values of all category

###Main Code
if __name__ == "__main__":
    create_training_data()
    rows,columns = training_data[0][0].shape

    random.shuffle(training_data)

    #Here X is used as a var for dataset
    #Here Y is used as a var for labels
    X = []
    Y = []

    for feature,label in training_data:
        X.append(feature)
        Y.append(label)

    X = np.array(X).reshape(-1,rows,columns,1)

    try:
        pickle_out = open(("X.pickle",'wb'))
        pickle.dump(X,pickle_out)
        pickle_out.close()
        pickle_out = open("Y.pickle",'wb')
        pickle.dump(Y,pickle_out)
        pickle_out.close()
    except Exception as e:
        print(Fore.RED + "Pickling failed")
        print(e)
    else:
        print(Fore.GREEN + "Pickled out successfully")
        print("object added: " + str(len(Y)))

    end = time.time()
    print(f"total time taken :{end-start}")
