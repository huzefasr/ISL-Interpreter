import cv2
cap = cv2.VideoCapture(-1)
ret,frame = cap.read()
while True:
	cv2.imshow("frame",frame)
