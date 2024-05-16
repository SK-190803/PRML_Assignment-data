#Compute following terms and print them:\\
#1. Difference of class wise means = ${m_1-m_2}$\\
#2. Total Within-class Scatter Matrix $S_W$\\
#3. Between-class Scatter Matrix $S_B$\\
#4. The EigenVectors of matrix $S_W^{-1}S_B$ corresponding to highest EigenValue\\
#5. For any input 2-D point, print its projection according to LDA.

import csv
import numpy as np

#mean1-mean0
def ComputeMeanDiff(X):
    # Extracting features and labels from given csv file
    features = X[:, :-1].astype(float)
    labels = X[:, -1].astype(float)

    # Creating array to store class means
    mean_0 = np.zeros(features.shape[1])
    mean_1 = np.zeros(features.shape[1])

    # Calculating mean vectors for both given classes
    count_0 = 0
    count_1 = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            mean_0 += features[i]
            count_0 += 1
        else:
            mean_1 += features[i]
            count_1 += 1
    # Calculating the mean difference array
    mean_0 /= count_0
    mean_1 /= count_1
    mean_diff = mean_1 - mean_0

    return mean_diff


def ComputeSW(X):
    # Extracting features and labels from the csv file
    features = X[:, :-1].astype(float)
    labels = X[:, -1]

    #Creating an within class Scatar matrix named SW
    SW = np.zeros((features.shape[1], features.shape[1]))

    # Calculating mean vectors for each class
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_samples = features[labels == label]
        # Calculating the mean vector for the current class
        mean_vector = np.mean(class_samples, axis=0)

        # Calculating the scatter matrix for the current class
        class_SW = np.zeros_like(SW)
        for sample in class_samples:
            deviation = sample - mean_vector
            class_SW += np.outer(deviation, deviation)

        SW += class_SW

    return SW


def ComputeSB(X):
    X_float = X.astype(float)
    
    # Computing the overall mean
    overall_mean = np.mean(X_float[:, :-1], axis=0)
    
    # Extracting class 0 and class 1 samples
    class_0_samples = X_float[X_float[:, -1] == 0][:, :-1]
    class_1_samples = X_float[X_float[:, -1] == 1][:, :-1]
    
    if len(class_0_samples) == 0 or len(class_1_samples) == 0:
        # If any class is empty, return a zero matrix
        return np.zeros((X.shape[1] - 1, X.shape[1] - 1))
    
    # Computing the mean of each class
    mean_0 = np.mean(class_0_samples, axis=0)
    mean_1 = np.mean(class_1_samples, axis=0)
    
    # Computing the deviation
    deviation_0 = mean_0 - overall_mean
    deviation_1 = mean_1 - overall_mean
    
    SB = np.outer(deviation_0, deviation_0) + np.outer(deviation_1, deviation_1)
    return SB


def GetLDAProjectionVector(X):
    SW = ComputeSW(X)
    SB = ComputeSB(X)
    # Computing the eigenvalues and eigenvectors of (SW^-1)*SB
    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.inv(SW), SB))
    # Finding the index of the eigenvector corresponding to the maximum eigenvalue
    max_index = np.nanargmax(eigen_values)
    
    #returning eigenvector having max eigenvalue
    return eigen_vectors[:, max_index]


def project(x, y, w):
    point = np.array([x, y])
    # Projecting the point onto the LDA projection vector w
    projection = np.dot(point, w)
    return projection


#########################################################
###################Helper Code###########################
#########################################################

X = np.empty((0, 3))
with open('E:\Pr\prmL_assignment\PA4\data.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for sample in csvFile:
        X = np.vstack((X, sample))

print(X)
print(X.shape)
# X Contains m samples each of formate (x,y) and class label 0.0 or 1.0

opt=int(input("Input your option (1-5): "))

match opt:
  case 1:
    meanDiff=ComputeMeanDiff(X)
    print(meanDiff)
  case 2:
    SW=ComputeSW(X)
    print(SW)
  case 3:
    SB=ComputeSB(X)
    print(SB)
  case 4:
    w=GetLDAProjectionVector(X)
    print(w)
  case 5:
    x=int(input("Input x dimension of a 2-dimensional point :"))
    y=int(input("Input y dimension of a 2-dimensional point:"))
    w=GetLDAProjectionVector(X)
    print(project(x,y,w))