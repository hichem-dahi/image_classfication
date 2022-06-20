import time
from sklearn.decomposition import PCA

import numpy as np
from sklearn.metrics import precision_score,recall_score
from LBP.LocalBinaryPattern import LocalBinaryPattern
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2 as cv
import os
from kmeans import Kmeans
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True, help="path to the testing images")
args = vars(ap.parse_args())

desc = LocalBinaryPattern(24, 2)
km_data = []
data = []
labels = []
ytrue = []
ypred = []
to_remove = {}
# s: Number of duplicate images
s = 0

#Eliminate duplicate images
IMAGE_SIZE = 320
sub_folders = ["yaleB"+ str(i) for i in range(11, 28)]

start_km = time.time()
for sub_folder in sub_folders:
    filenames = []
    km_data = []
    for imagePath in paths.list_images("images/training/" + sub_folder):
        if imagePath != None:
            image = cv.imread(imagePath)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gray = cv.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
            km_data.append(gray)
            filenames.append(imagePath)
    if km_data != []:
       km_data = np.array(km_data)
       km = Kmeans(km_data, filenames=filenames)
       to_remove[sub_folder] = km.find_to_remove_files()
end_km = time.time()
print(f"Runtime of the program is {end_km - start_km}")


for sub_folder in to_remove:
    print(len(to_remove[sub_folder]))
    s += len(to_remove[sub_folder])
print("Number of duplicate images: "+ str(s)) 


start = time.time()
for imagePath in paths.list_images(args["training"]):
    if imagePath not in to_remove[imagePath.split(os.path.sep)[-2]]:
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(hist)
end = time.time()

model = LinearSVC(C=100.0, random_state=42)

# Boolean to use PCA or not
usePca = False

print(f"Runtime of the program is {end - start}")

if usePca == False:
    model.fit(data, labels) 
    # Testing - Classification
    for imagePath in paths.list_images(args["testing"]):
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
        label = imagePath.split(os.path.sep)[-1]
        label = label[0:7]
        ytrue.append(label) 
        ypred.append(prediction[0])
        #cv.putText(image, prediction[0], (10, 30), cv.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
        #cv.imshow("Image", image)
        #cv.waitKey(100)
else:
    pca = PCA(2)
    data_reduced = pca.fit_transform(data)
    # Use data_reduced for pca option
    model.fit(data_reduced, labels) 
    # Testing - PCA + Classification
    for image, label in zip(data_reduced, labels):
        prediction = model.predict(np.array(image).reshape(1,-1))
        ytrue.append(label) 
        ypred.append(prediction[0])


precision = precision_score(ytrue, ypred, average='micro')
rappel = recall_score(ytrue, ypred, average='micro')
print(f'precision = {precision}%')
print(f'recall = {rappel}%')