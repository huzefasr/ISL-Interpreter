import os
import cv2

def save_images(path_og):
	files = os.listdir(path_og)
	ext = ".png"
	img = cv2.imread("/home/chintan/Documents/Indian-Sign-Language-Interpreter/1.png")
	new_file = []
	if files == []:
		name = 0
	else:
		try:
			for file in files:
				file = file.split('.')
				new_file.append(int(file[0]))
			new_file.sort()
			name = new_file[-1]
			for num in range(10):
				name = name+1
				imgname = str(name)+'.png'
				imgname = os.path.join(path_og,imgname)
				cv2.imwrite(imgname,img)
				print(imgname)
		except:
			print("Error"+e)
	cv2.destroyAllWindows()

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

########
path_og = path()
print(path_og)

save_images(path_og)
