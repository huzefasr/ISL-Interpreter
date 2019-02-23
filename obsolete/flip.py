import os
import cv2
import sys
def path_finder(char):
    path = os.getcwd()
    images = []

    path = os.path.join(path,'dataset')
    path = os.path.join(path,char)
    last = sorted(os.listdir(path))
    for item in last:
        if item.endswith('.png'):
            name = item.split('.')
            images.append(int(name[0]))
    images = sorted(images)
    last = images[-1]
    return path,images,last

####CODE
path,images,last = path_finder(sys.argv[1])
os.chdir(path)
for img in images:
    last= last+1
    img = cv2.imread(str(img)+'.png')
    cv2.imshow("img",img)
    flip = cv2.flip(img,1)
    cv2.imwrite("{}".format(last)+".png",flip)
    print("successfully")
