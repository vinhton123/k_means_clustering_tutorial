import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

# Read in and convert an image file into a 3D array of pixels

# If using jpg files, make sure to cast your np array as type integer
# as each pixel rgb value uses 0 - 255 and imshow expects a float (0 - 1)
image_3D_array = plt.imread(("smiley.jpg"))
plt.imshow(image_3D_array.astype(np.uint8))

# If using png files, it already uses floats (0 - 1) for its pixels
# so no type casting is necessary
image_3D_array_example2 = plt.imread("portrait.png")
plt.imshow(image_3D_array_example2)

# Check each dimension of the 3D array of pixels representing the image
# 1st element = width dimension of pixel
# 2nd element = height dimension of pixel
# 3rd element = rgb value of pixel

width_px_dim = image_3D_array.shape[0]
height_px_dim = image_3D_array.shape[1]
rgb_px_dim = image_3D_array.shape[2]

print('width dimension: ' + str(width_px_dim))
print('height dimension: ' + str(height_px_dim))
print('rgb value dimension (3 for red (0 - 255), green (0 - 255), and blue (0 - 255)): ' + str(rgb_px_dim))

# Reshape the 3D image array into a 2D array in order to process it using K Means Clustering
image_2D_array = image_3D_array.reshape(width_px_dim * height_px_dim, rgb_px_dim)
loc_px_dim_2D = image_2D_array.shape[0]
rgb_px_dim_2D = image_2D_array.shape[1]
print(image_2D_array[0][1])
print('pixel location dimension: ' + str(loc_px_dim_2D))
print('rgb value dimension (3 for red (0 - 255), green (0 - 255), and blue (0 - 255)): ' + str(rgb_px_dim_2D))

# Perform K Means Clustering on this 2D Array

# Step 1:
# The rgb value of the image is the important part for image processing (what you see)
# so get the number of features we are interested in are the original dimensions of the rgb
N_FEATURES = rgb_px_dim
print(N_FEATURES)

# This should print 3 for red, green, and blue value for jpg files (or 4 if you are using png)

# Step 2:
# Choose the number of clusters you want
# In this case, this number you choose this will be the number of distinct color categories
# on your resulting image (I will choose 4 arbitrarily)
# The colors of the categories are the value in the middle of a clusters that minimizes
# the distance of each value in that cluster
# (or in this case, the best color that is the average for each categories) 
N_CLUSTERS = 3


# Step 3
# Use the ski-kit library to perform the K Means Clustering algorithm on this array of data
centroids = KMeans(n_clusters=N_CLUSTERS).fit(image_2D_array)
centroids_array = centroids.cluster_centers_
centroid_labels = centroids.labels_


# Step 4
# Print out the resulting iamge

# Convert the 2D array of centroids back into the original 3D image array
# to prepare to print the processed image
centroids_3D_array = centroids_array[centroid_labels].reshape(
    width_px_dim, height_px_dim, rgb_px_dim)
plt.imshow(centroids_3D_array.astype(np.uint8))

# for png: plt.imshow(centroids_3D_array)