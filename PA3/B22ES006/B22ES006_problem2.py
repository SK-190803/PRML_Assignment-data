#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier



'''Task-1:Data Preprocessing'''
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Extract the images and corresponding target labels
X_faces = lfw_people.images
y = lfw_people.target

# Split the dataset into training and testing sets with 80:20 split ratio
X_train, X_test, y_train, y_test = train_test_split(X_faces, y, test_size=0.2, random_state=84)

# Print the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Task-2: Eigenfaces Implementation using PCA
# Define PCA class
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        # Calculate the covariance matrix of the training data
        cov_matrix = np.cov(X.T)
        
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort the eigenvectors based on their corresponding eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Choose the top n_components eigenvectors as principal components
        self.components = sorted_eigenvectors[:, :self.n_components]
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[sorted_indices][:self.n_components] / total_variance

        # Set the attribute n_components_ to the actual number of components
        self.n_components_ = self.components.shape[1]

    def transform(self, X):
        # Project the data onto the principal components
        return np.dot(X, self.components)


# Reshape the 2D images into 1D vectors
X_train_flat = X_train.reshape((X_train.shape[0], -1))

# Normalize pixel values to range [0, 1]
X_train_flat = X_train_flat / 255.0

# Initialize PCA with the desired number of components
n_components = 200  # 
pca = PCA(n_components=n_components)

# Fit PCA on the training data
pca.fit(X_train_flat)

# Transform the training data into the reduced-dimensional space
X_train_pca = pca.transform(X_train_flat)

# Plot the explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()

# Print explained variance ratio for each component
print("Explained Variance Ratio for each Component:")
for i, explained_variance in enumerate(pca.explained_variance_ratio_[:10]):
    print(f"Component {i+1}: {explained_variance:.3f}")

# Choose the number of components to retain based on the explained variance ratio
print("\nNumber of components selected based on explained variance ratio:", pca.n_components_)

'''Task-3: Model Training'''
class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, k=None):
        if k is None:
            k = self.n_neighbors

        y_pred = np.zeros(X_test.shape[0])

        for i, x_test in enumerate(X_test):
            # Compute distances between current test point and all training points
            distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))

            # Sort by distance and return indices of the first k neighbors
            nearest_indices = np.argsort(distances)[:k]

            # Get the labels of the k nearest neighbor training samples
            k_nearest_labels = self.y_train[nearest_indices]

            # Predict the class of the current test sample based on majority vote
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

        return y_pred

# Initialize KNN classifier
knn_classifier = KNN(n_neighbors=5)

# Train the classifier using the transformed training data
knn_classifier.fit(X_train_pca, y_train)

# Transform the testing data into the reduced-dimensional space
X_test_flat = X_test.reshape((X_test.shape[0], -1))
X_test_flat = X_test_flat / 255.0  # Normalize pixel values to range [0, 1]
X_test_pca = pca.transform(X_test_flat)

# Define Linear Regression class
class LinearRegressionClassifier:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_test):
        return np.round(self.model.predict(X_test)).astype(int)

# Define Decision Tree Classifier class
class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Train and evaluate Linear Regression classifier
linear_regression_classifier = LinearRegressionClassifier()
linear_regression_classifier.fit(X_train_pca, y_train)
y_pred_lr = linear_regression_classifier.predict(X_test_pca)

# Train and evaluate Decision Tree classifier
decision_tree_classifier = DecisionTree()
decision_tree_classifier.fit(X_train_pca, y_train)
y_pred_dt = decision_tree_classifier.predict(X_test_pca)


print("-------------------------------------------------------------------------------------")

'''Task-4:Model Evaluation'''
# Predict labels for the transformed testing data
y_pred = knn_classifier.predict(X_test_pca)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("KNN Accuracy:", accuracy)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Linear Regression Classifier Accuracy:", accuracy_lr)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Classifier Accuracy:", accuracy_dt)
print("----------------------------------------------------------------------------------------")
# Plotting comparison graph
classifiers = ['KNN', 'Linear Regression', 'Decision Tree']
accuracies = [accuracy, accuracy_lr, accuracy_dt]

plt.bar(classifiers, accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Comparison of Classifier Accuracies')
plt.show()

print("----------------------------------------------------------------------------------------")
print("Since,Use of KNN Classifier with PCA gives the best accuracy!-It is being used for futher analysis")
print("                                                                                        ")

# Visualize a subset of Eigenfaces
# Obtain height and width from the shape of the images
h, w = X_train.shape[1:]
# Visualize a subset of Eigenfaces
n_eigenfaces_to_visualize = 10
fig, axes = plt.subplots(1, n_eigenfaces_to_visualize, figsize=(20, 4))

for i, ax in enumerate(axes):
    eigenface = pca.components[:, i].reshape((h, w))
    ax.imshow(eigenface, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Eigenface {i+1}')

plt.show()

'''Task5:-Experiment with different values of ncomponents'''
# List of n_components to experiment with
n_components_values = [50, 100, 150, 200, 250]

# Dictionary to store accuracy for each value of n_components
accuracy_scores = {}

for n_components in n_components_values:
    # Initialize PCA with the current value of n_components
    pca = PCA(n_components=n_components)

    # Fit PCA on the training data
    pca.fit(X_train_flat)

    # Transform training and testing data into the reduced-dimensional space
    X_train_pca = pca.transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    # Train KNN classifier using transformed training data
    knn_classifier.fit(X_train_pca, y_train)

    # Predict labels for the transformed testing data
    y_pred = knn_classifier.predict(X_test_pca)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    # Store accuracy for the current value of n_components
    accuracy_scores[n_components] = accuracy

# Print accuracy scores for each value of n_components
print("Accuracy Scores for Different Values of n_components:")
for n_components, accuracy in accuracy_scores.items():
    print(f"n_components={n_components}: Accuracy={accuracy:.4f}")

