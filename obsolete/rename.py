import os
import sys
from colorama import Fore
path = os.getcwd()
path = os.path.join(path,'dataset')
j = 1
i = 1
print(len(sys.argv))
#print(sys.argv[0])
#print(sys.argv[1])
#print(sys.argv[2])
while j < len(sys.argv):
    print("input folder is {}".format(sys.argv[j]))
    path_char = os.path.join(path,'{}'.format(sys.argv[j]))
    files = os.listdir(path_char)
    print(sorted(files))
    for file in sorted(files):
        print(file)
        os.rename(os.path.join(path_char, file), os.path.join(path_char, str(i)+'.png'))
        i = i+1
        print(file)
    print(Fore.GREEN+str(sys.argv[j]) + " Has been sorted" )
    j = j+1

path = os.getcwd()
path = os.path.join(path,'dataset')

for letter in len(sys.argv)-1:
    print("input folder is {}".format(sys.argv[1]))
    path = os.path.join(path,'{}'.format(sys.argv[1]))
    files = os.listdir(path)
    i = 1
    for file in sorted(files):
        os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.png'))
        i = i+1
