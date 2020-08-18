import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# declare constants
N_SAMPLES = 100
SECOND_HALF = 50
N_FEATURES = 3
N_CLUSTERS = 4

# generate sample data
data_array = -2 * np.random.rand(N_SAMPLES,N_FEATURES)
data_array[SECOND_HALF:N_SAMPLES, :] = 1 + 2 * np.random.rand(SECOND_HALF,N_FEATURES)


# generate centroids
centroids = KMeans(n_clusters=N_CLUSTERS).fit(data_array)
centroids_array = centroids.cluster_centers_
print(centroids.labels_)

# visualize data
fig = plt.figure()
plot = fig.add_subplot(111, projection='3d')
plot.scatter(data_array[ : , 0], data_array[ :, 1], data_array[ : , 2], s = 20, color = "blue", marker='o')

for centroid in centroids_array:
    feature1_mean = centroid[0]
    feature2_mean = centroid[1]
    feature3_mean = centroid[2]
    plot.scatter(feature1_mean, feature2_mean, feature3_mean, s=400, color = "red", marker = 'o')

plt.show()
