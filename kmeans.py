from time import time
from cv2 import kmeans
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
import json
from scipy.spatial.distance import cdist
import cv2 as cv
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
class Kmeans:
    def __init__(self, data, filenames) -> None:
        self.data = data
        self.filenames = filenames

        pass

    def plot_data_in_clusters(self, data_reduced, kmeans, idx=None, show_centroids=True):
        marker_size = 7

         # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = data_reduced[:, 0].min(), data_reduced[:, 0].max()
        y_min, y_max = data_reduced[:, 1].min(), data_reduced[:, 1].max()

          # Step size of the mesh. Decrease to increase the quality of the VQ.
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        h = float((x_max - x_min)/100)

        PADDING = h * marker_size
        x_min, x_max = x_min - PADDING, x_max + PADDING/2
        y_min, y_max = y_min - PADDING, y_max + PADDING

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plt.figure(2)
        # plt.clf()
        plt.imshow(Z, interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(data_reduced[:, 0], data_reduced[:, 1], 'k.', markersize=marker_size)

        if show_centroids:
            markers = ["o", "1",]
            # Plot the centroids as a white X
            centroids = kmeans.cluster_centers_

        for id in range(len(centroids)):
            c = centroids[id]
            print(len(centroids))
            plt.scatter(c[0], c[1], marker=markers[1], s=150, linewidths=marker_size,
            color="w", zorder=10)
        if idx:
            for id in idx:
                plt.scatter(data_reduced[id, 0], data_reduced[id, 1], marker="x",
                    s=150, linewidths=marker_size,
                    color="w", zorder=10)
        plt.title("KMeans clustering")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        pass

    def get_pca_reduced(self,X_fm):
        X_fm_flat = X_fm.reshape(X_fm.shape[0], -1)
        print("shape "+ str(X_fm_flat.shape))
        pca = PCA(2)
        if (X_fm.shape[0] == 1):
            pca = PCA(1)
        X_fm_reduced = pca.fit_transform(X_fm_flat)
        return X_fm_reduced, pca

    def get_clusters(self, X, K):
        kmeans = KMeans(n_clusters=K, random_state=0)
        X_clusters = kmeans.fit(X)
        return X_clusters, kmeans
    
    def get_cluster_groups(self, K):
        # Use these lines to use PCA
        data_reshaped = preprocessing.normalize(self.data.reshape(self.data.shape[0], -1))
        data_reduced, pca1 = self.get_pca_reduced(data_reshaped)
        
        data_clusters_reduced, kmeans_reduced = self.get_clusters(data_reduced, K)
        groups_fn = {}
        for file, cluster in zip(self.filenames, kmeans_reduced.labels_):
            if cluster not in groups_fn.keys():
                groups_fn[cluster] = []
                groups_fn[cluster].append(file)
            else:
                groups_fn[cluster].append(file)
        #print(groups_fn)
        print("------------------")
        #print(groups_fn[1])
        IMAGE_SIZE = 320
        data_clusters_final = []
        for group in groups_fn:
            data_cluster = []
            for imagePath in groups_fn[group]:
                image = cv.imread(imagePath)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                gray = cv.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
                data_cluster.append(gray)
            data_cluster = np.array(data_cluster)
            data_cluster_reshaped = preprocessing.normalize(data_cluster.reshape(data_cluster.shape[0], -1))
            data_cluster_reduced, pca3 = self.get_pca_reduced(data_cluster_reshaped)
            data_clusters_final.append(data_cluster_reduced)
        return groups_fn, data_clusters_final, kmeans_reduced, data_reduced
            
    def find_duplicates_to_remove(self, X_train_pca, threshold=0.05):
        # Calculate distances of all points
        distances = cdist(X_train_pca, X_train_pca)
        # Find duplicates (very similar images)
        dupes = [np.array(np.where(distances[id] < threshold)).reshape(-1).tolist() \
        for id in range(distances.shape[0])]
        #extreme_vals = [np.array(np.where(distances[id] > 0.999)).reshape(-1).tolist() \
        #for id in range(distances.shape[0])] 
        #dupes += extreme_vals
        to_remove = set()
        for d in dupes:
            if len(d) > 1:
                for id in range(1, len(d)):
                    to_remove.add(d[id])
        print("Found {} duplicates".format(len(to_remove)))
        return to_remove
    
    def find_to_remove_files(self):
        to_remove_fn = []
        to_remove_toplt = []
        groups_fn, cluster_groups, kmeans, data_reduced= self.get_cluster_groups(K=9)
        for cluster, cluster_group in zip(groups_fn.keys(), cluster_groups):
            to_remove = self.find_duplicates_to_remove(cluster_group)
            to_remove_toplt += to_remove
            #to_remove_fn.append([groups_fn[cluster][i] for i in to_remove])
            to_remove_fn = to_remove_fn + [groups_fn[cluster][i] for i in to_remove]
        # Plot data
        print(to_remove_toplt)
        self.plot_data_in_clusters(data_reduced, kmeans=kmeans)
        self.plot_data_in_clusters(data_reduced, kmeans=kmeans,idx=to_remove_toplt)
        return to_remove_fn
"""
data = []
filenames = []
IMAGE_SIZE = 320

for imagePath in paths.list_images("images/training/yaleB27"):
    if imagePath != None:
        image = cv.imread(imagePath)
        image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        data.append(image)
        filenames.append(imagePath)  
data = np.array(data)
km = Kmeans(data, filenames=filenames)  
"""
#Detection duplicate images
data = []
filenames = []

to_remove = []
IMAGE_SIZE = 320

#Detection duplicate images
for imagePath in paths.list_images("images/training/yaleB11"):
    if imagePath != None:
        image = cv.imread(imagePath)
        image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        data.append(image)
        filenames.append(imagePath)
if data != []:
    data = np.array(data)
    km = Kmeans(data, filenames=filenames)
    to_remove = km.find_to_remove_files()
print(to_remove)

for imagePath in to_remove:
    image = cv.imread(imagePath)
    cv.imshow("image", image)
    cv.waitKey(1000)