import numpy as np
from sklearn.cluster import k_means
from imutils import paths
import cv2 as cv
from kmeans import Kmeans

data = labels = ypred = ytrue = km_data  = []
to_remove = {}
IMAGE_SIZE = 320

#Detection duplicate images
IMAGE_SIZE = 320
sub_folders = ["yaleB"+ str(i) for i in range(11, 28)]

for sub_folder in sub_folders:
    filenames = []
    data = []
    for imagePath in paths.list_images("images/training/" + sub_folder):
        if imagePath != None:
            image = cv.imread(imagePath)
            image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            data.append(image)
            filenames.append(imagePath)
    if data != []:
       data = np.array(data)
       km = Kmeans(data, filenames=filenames)
       to_remove[sub_folder] = km.find_to_remove_files()
print(to_remove)

    