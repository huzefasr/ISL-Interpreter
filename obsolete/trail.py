import os
path = os.getcwd()
path = os.path.join(path,'dataset')
category = os.listdir(path)
category = category.sort()
print(category)
