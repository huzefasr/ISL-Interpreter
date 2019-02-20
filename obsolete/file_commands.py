'''
This application is used to perform different
operations on the dataset created
below is a list of command available
rename
check extra
'''
import os
def file_command():
    path = os.getcwd()
    path = os.path.join(path,'dataset')
    #print(path)
    data = os.listdir(path)
    data = sorted(data)
    #print(str(len(data)) + " folders found")-->[a,b.....,z]
    try:
        for char in os.listdir(path):
            i = 1
            path_old = os.path.join(path,char)
            path_new = os.path.join(path,char+"1")
            os.makedirs(path_new,exist_ok=True)
            os.chdir(path_old)
            print(path_old)
            for file in path_old:
                os.rename(file,str(i)+'.png')
                i = i + 1
                #print("file renamed to " + file)
    except Exception as e:
        print(e)

if __name__=="__main__":
    file_command()
