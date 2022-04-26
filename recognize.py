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

#Detection duplicate images
IMAGE_SIZE = 320
sub_folders = ["yaleB"+ str(i) for i in range(11, 28)]

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

for imagePath in paths.list_images(args["training"]):
    if imagePath not in to_remove[imagePath.split(os.path.sep)[-2]]:
        image = cv.imread(imagePath)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(hist)
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

#Testing
for imagePath in paths.list_images(args["testing"]):
    image = cv.imread(imagePath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    label = imagePath.split(os.path.sep)[-1]
    label = label[0:7]
    ytrue.append(label) 
    ypred.append(prediction[0])
    cv.putText(image, prediction[0], (10, 30), cv.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    cv.imshow("Image", image)
    cv.waitKey(1000)


print("true")
print(ytrue)
print("pred")
print(ypred)
precision = precision_score(ytrue, ypred, average='micro')
rappel = recall_score(ytrue, ypred, average='micro')
print(f'precision = {precision}%')
print(f'recall = {rappel}%')