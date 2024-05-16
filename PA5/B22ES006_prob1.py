#importing required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v2 as imageio

#Task (a) and (b)
# computeCentroid function, that takes n 3-dimensional features and returns their mean.
def computeCentroid(X):
    return np.mean(X, axis=0)

#mykmeans from scratch that takes data matrix X of size m×3 where m is the number of pixels in the image and the number of clusters k. It returns the cluster centers using the k-means algorithm.
def mykmeans(X, k, max_iter=100, tol=1e-6):
    m, n = X.shape

    # Initializing cluster centroids randomly
    centroids = X[np.random.choice(m, k, replace=False), :]

    # Looping until convergence or maximum iterations reached
    for iter in range(max_iter):
        # Computing distances between data points and centroids
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=-1))

        # Assigning data points to closest centroids
        cluster_assignments = np.argmin(distances, axis=1)

        # Updating centroids by taking the mean of assigned data points
        new_centroids = np.zeros((k, n))
        for i in range(k):
            cluster_points = X[cluster_assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)

        # Checking convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids

#Task (c)
image_url = 'https://raw.githubusercontent.com/the-punisher-29/PRML_Assignment-data/main/test.png'
# Loading the image
image = imageio.imread(image_url)
print("Image shape:", image.shape)
# Displaying the image(RGB form)
cv2.imshow('Org Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2_imshow(img1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
# Reshaping the image data to (m, 3) where m is the number of pixels
h, w, c = image.shape
X = image.reshape(-1, c)
#values of k
k_values = [2, 4, 8, 16]

# Compressing the image using k-means defined froms scratch for different values of k
compressed_images = []
for k in k_values:
    centroids = mykmeans(X, k)
    compressed_image = centroids[np.argmin(np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=-1)), axis=1)]
    compressed_image = compressed_image.reshape(h, w, 3).astype(np.uint8)
    compressed_images.append(compressed_image)

# Displaying the compressed images
print('Image Compression using kmeans defined from scratch')
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i in range(len(compressed_images)):
    ax = axs.flatten()[i]
    compressed_image = compressed_images[i]
    ax.imshow(compressed_image)
    ax.set_title(f'k = {k_values[i]}')
    ax.axis('off')

plt.show()
print('-------------------------------------------------------------------------------------------------------------')

#Task (d)
#compressing the image using ready made-kmeans from sklearn library
from sklearn.cluster import KMeans
'''cv2.imshow("RGB_Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# Reshaping the image
h, w, c = image.shape
X = image.reshape(-1, c)
#values of k
k_values = [2, 4, 8, 16]

# Compressing the image using scikit-learn k-means
compressed_images = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = compressed_image.reshape(h, w, c).astype(np.uint8)
    compressed_images.append(compressed_image)

# Displaying the compressed images
print('Image Compression using sklearn kmeans ')
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
for i, (ax, compressed_image) in enumerate(zip(axs, compressed_images)):
    ax.imshow(compressed_image)
    ax.set_title(f'k = {k_values[i]}')
    ax.axis('off')

plt.show()
#squared error for a given k
def calculate_squared_error(X, k):
    kmeans = KMeans(n_clusters=k, random_state=29, n_init=10).fit(X)
    squared_error = kmeans.inertia_
    return squared_error
#  squared error for each k
squared_errors = [calculate_squared_error(X, k) for k in k_values]
# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, squared_errors, marker='o', linestyle='-')
plt.title('Squared Error vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Squared Error')
plt.grid(True)
plt.show()

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def spatial_distance(x1, y1, x2, y2):
    #Euclidean distance between two points
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def spatial_coherence_distance(pixel1, pixel2, spatial_weight=0.5):
    # Calculating spatial coherence distance between two pixels
    color_distance = np.linalg.norm(pixel1 - pixel2)
    spatial_distance = spatial_weight * spatial_distance(pixel1[0], pixel1[1], pixel2[0], pixel2[1])
    return color_distance + spatial_distance

def compress_image_with_spatial_coherence(image, k, spatial_weight=0.5):
    h, w, c = image.shape
    X = image.reshape(-1, c)
    
    # Generating spatial coordinates
    spatial_coordinates = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    
    # Combining color features and spatial coordinates
    features = np.concatenate((X, spatial_coordinates), axis=1)
    
    # Performing K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=29, n_init=10).fit(features)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_[:, :c]  # Exclude spatial coordinates from cluster centers
    
    # Reconstructing compressed image
    compressed_image = np.zeros_like(X)
    for i, label in enumerate(labels):
        compressed_image[i] = cluster_centers[label]
    compressed_image = compressed_image.reshape(h, w, c).astype(np.uint8)
    
    return compressed_image

#image to BGR color space
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
compressed_image = compress_image_with_spatial_coherence(image_bgr, k=16, spatial_weight=0.5)

#original and compressed images
plt.figure(figsize=(10, 5))
cv2.imshow("Original Image",image_bgr)
cv2.imshow("Spatial Coherence",compressed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()