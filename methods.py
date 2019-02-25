'''
This file contains methods that have been tried to acomplish the goal of abstracting
the hand from the background
'''
import cv2
import numpy as np

#method 1 - inrange
#method 2 - pyrMeanShiftFiltering
#method 3 - calcBackProject

def method_inrange(flip,lowerBoundary,upperBoundary):
    hsv_flip = cv2.cvtColor(flip,cv2.COLOR_BGR2HSV)
    # Defining skin color parameters to detect skin
    #lowerBoundary = np.array([0,40,30],dtype="uint8")# 0 40 30
    #upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(hsv_flip, lowerBoundary, upperBoundary)
    #method_pymean(skinMask)
    # Defining kernenls
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # Performing image filtering
    skinMask = cv2.filter2D(skinMask,-1,15)
    skinMask = cv2.erode(skinMask, kernel5,3)
    skinMask = cv2.dilate(skinMask, kernel5,5)
    cv2.imshow("skinmask",skinMask)

    return skinMask

def method_pymean(flip):
    #flip2 = cv2.resize(flip,(200,200))
    rgb_flip = cv2.cvtColor(flip,cv2.COLOR_HSV2RGB)
    rgb_flip = cv2.pyrMeanShiftFiltering(rgb_flip,21,51)
    gray = cv2.cvtColor(rgb_flip,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("gray",thresh)
    return thresh

def method_backproject(hsv_flip,roi_hist):
	'''
	This method is used to convert the hsv image into a binary image using histogram
	'''
	#hsv_flip = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	mask = cv2.calcBackProject([hsv_flip],[0,1],roi_hist,[0,180,0,256],1)


	# Defining the kernel used later
	kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))

	img = cv2.filter2D(mask,-1,kernel15) # 9 or 11 works
	img = cv2.bilateralFilter(img,15,75,50) #
	img = cv2.GaussianBlur(img,(5,5), 0)
    
	_,thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
	#img = cv2.dilate(thresh,kernel3,iterations = 1)
	#cv2.imshow("final",img)
	return thresh
