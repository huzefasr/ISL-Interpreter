import os
import cv2
def path():
	path = os.getcwd()
	path = os.path.join(path,'dataset')
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

def save_images(path_og):
	cap = cv2.VideoCapture(0)
	keyc = False
	keyx = False
	while True:
		_,frame = cap.read()
		key = cv2.waitKey(1)
		flip = cv2.flip(frame,1)
		y,x,c = flip.shape
		x1 = int(x/2)
		y1 = int(y/4)
		x2 = x1+300
		y2 = y1+200

		flip = convert(flip)

		rect = cv2.rectangle(flip, (x1,y1), (x2,y2), (255,0,0), 1)
		cv2.imshow("flip",flip)
		if key == ord('c'):
			keyc = True
		if keyc:
			files = os.listdir(path_og)
			ext = ".png"
			if files == []:
				name = 0
			else:
				name,ext = os.path.splitext(files[-1])
				name = int(name)
			for n in range(1200):
				_,frame = cap.read()
				flip = cv2.flip(frame,1)
				flip  = flip[y1:y2,x1:x2]
				#flip = convert(flip)
				name = name+1
				imgname = str(name)+'.png'
				imgname = os.path.join(path_og,imgname)
				flip = cv2.resize(flip,(50,50))
				cv2.imwrite(imgname,flip)
				name = name+1
				print(path_og)
				normal = cv2.flip(flip,1)
				imgname = os.path.join(path_og,imgname)
				cv2.imwrite(imgname,normal)
				print(path_og)
			keyc = False
		if key == ord('x'):
			break

	cap.release()
	cv2.destroyAllWindows()

path_og = path()
save_images(path_og)
