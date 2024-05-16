#Task 1a and 1b
#importing rtequired libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Loading the Iris dataset
iris = datasets.load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']]
y = iris.target
# Selecting only 'setosa' and 'versicolor' classes
mask = y != 2
X = X[mask]
y = y[mask]
# Normalizing the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=29)

# Training (LinearSVC)
model = LinearSVC(dual=False)
model.fit(X_train, y_train)

#function for plotting decision boundary
def plot_decision_boundary(ax, X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.viridis, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.viridis, edgecolors='red',label=['Class 0', 'Class 1'])
    ax.set_xlabel('Petal length (cm)')
    ax.set_ylabel('Petal width (cm)')
    ax.set_title(title)

# Plotting decision boundaries with training and test data
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plot_decision_boundary(axs[0], X_train, y_train, model, 'Decision Boundary on Training Data')
plot_decision_boundary(axs[1], X_test, y_test, model, 'Decision Boundary with Test Data')
plt.tight_layout()
plt.show()

#Task 2(a),2(b),2(c),2(d)

#a
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generating synthetic dataset with make_moons
X, y = make_moons(n_samples=500, noise=0.05, random_state=29)
# visualzing the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.inferno, edgecolors='red',label=['Class 0', 'Class 1'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

#b
#plotting decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = .02  #step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.plasma, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.plasma, edgecolors='red',label=['Class 0', 'Class 1'])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Linear SVM
linear_svc = SVC(kernel='linear')
linear_svc.fit(X, y)
plot_decision_boundary(linear_svc, X, y, 'Linear SVM')

# Polynomial SVM
poly_svc = SVC(kernel='poly', degree=3)# taking polynomial of degree 3
poly_svc.fit(X, y)
plot_decision_boundary(poly_svc, X, y, 'Polynomial SVM (Degree 3)')


# RBF SVM
rbf_svc = SVC(kernel='rbf', gamma='auto')
rbf_svc.fit(X, y)
plot_decision_boundary(rbf_svc, X, y, 'RBF SVM')

#c and d
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
# Parameter grid definition
param_distribs = {
    'C': reciprocal(20, 200000),
    'gamma': reciprocal(0.0001, 0.1),
}
# RandomizedSearchCV object creation
random_search = RandomizedSearchCV(SVC(kernel='rbf'), param_distributions=param_distribs, n_iter=10, cv=5, scoring='accuracy')
# Performing random search
random_search.fit(X, y)
# Best hyperparameters
print("Best hyperparameters:", random_search.best_params_)
# Best model
best_rbf_svc = random_search.best_estimator_
# Decision boundary of the best model
plot_decision_boundary(best_rbf_svc, X, y, 'RBF SVM with Best Hyperparameters')