import os
import sys
import numpy as np

#Generation of synthetic data;
def generate_dataset(num_samples, seed=None):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate random weights as floats
    weights = np.random.uniform(-1, 1, size=5).astype(np.float32)  # Including bias term

    # Generate random input vectors
    X = np.random.randint(0, 10, size=(num_samples, 4), dtype=np.int32)

    # Calculate f(x) for each input vector
    f_values = np.dot(X, weights[1:]) + weights[0]

    # Assign labels based on f(x) using thresholding
    y = np.where(f_values >= 0, 1, 0)

    print("Generated Weights (w0, w1, w2, w3, w4):", weights)  # Print generated weights

    return X, y, weights

def save_to_file(X, y, filename):
    with open(filename, 'w') as file:
        file.write(f"{len(X)}\n")
        for i in range(len(X)):
            line = ' '.join(map(str, X[i].tolist() + [y[i]])) + '\n'
            file.write(line)

def save_features_to_file(X, filename):
    with open(filename, 'w') as file:
        file.write(f"{len(X)}\n")
        for i in range(len(X)):
            line = ' '.join(map(str, X[i].tolist())) + '\n'
            file.write(line)

def split_dataset(X, y, train_percentage):
    num_samples = len(X)
    num_train_samples = int(num_samples * train_percentage / 100)

    # Shuffle the dataset
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the dataset
    X_train = X_shuffled[:num_train_samples]
    y_train = y_shuffled[:num_train_samples]
    X_test = X_shuffled[num_train_samples:]
    y_test = y_shuffled[num_train_samples:]

    return X_train, y_train, X_test, y_test

# Get the directory of the Python script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Generate synthetic dataset
num_samples = 1000
seed = 39  # Set the seed value
X, y, weights = generate_dataset(num_samples, seed)

# Save data to file
save_to_file(X, y, os.path.join(current_directory, "data.txt"))

# Generate 20%, 50%, and 70% subsets of the data
subsets = [20, 50, 70]
for subset_percentage in subsets:
    subset_size = int(num_samples * subset_percentage / 100)
    indices = np.random.choice(num_samples, subset_size, replace=False)
    subset_X = X[indices]
    subset_y = y[indices]
    save_to_file(subset_X, subset_y, os.path.join(current_directory, f"m{subset_percentage}.txt"))

# Definition of load_data function to load data from files
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_samples = int(lines[0])
        data = np.zeros((num_samples, 4), dtype=int)
        labels = np.zeros(num_samples, dtype=int)
        for i, line in enumerate(lines[1:]):
            values = line.strip().split()
            data[i] = list(map(int, values[:4]))
            labels[i] = int(values[4])
    return data, labels

# Generate train and test files for each subset
for subset_percentage in subsets:
    # Load the subset data from files m20.txt, m50.txt, and m70.txt
    subset_X, subset_y = load_data(os.path.join(current_directory, f"m{subset_percentage}.txt"))

    # Split the subset data into train and test sets with an 80:20 split
    train_X, train_y, test_X, test_y = split_dataset(subset_X, subset_y, 80)

    # Save the train and test data into separate files
    save_to_file(train_X, train_y, os.path.join(current_directory, f"train_m{subset_percentage}.txt"))
    save_to_file(test_X, test_y, os.path.join(current_directory, f"test_m{subset_percentage}.txt"))



class Perceptron:
    def __init__(self, learning_rate=0.01, num_epochs=700):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None

    def train(self, X_train, y_train):
        # Normalize data
        X_train_normalized = self.normalize_data(X_train)

        # Initialize weights
        num_features = X_train_normalized.shape[1]
        self.weights = np.zeros(num_features + 1)  # Additional weight for bias term

        # Train the perceptron
        for _ in range(self.num_epochs):
            for i in range(len(X_train_normalized)):
                prediction = self.predict(X_train_normalized[i])
                error = y_train[i] - prediction
                self.weights[:-1] += self.learning_rate * error * X_train_normalized[i]
                self.weights[-1] += self.learning_rate * error  # Update bias term

    def predict(self, x):
        return 1 if np.dot(x, self.weights[:-1]) + self.weights[-1] >= 0 else 0

    def normalize_data(self, X):
        return X / np.linalg.norm(X)

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_samples = int(lines[0])
        data = np.zeros((num_samples, 4))
        labels = np.zeros(num_samples)
        for i, line in enumerate(lines[1:]):
            values = line.strip().split()
            data[i] = list(map(int, values[:4]))
            labels[i] = int(values[4])
    return data, labels

def save_weights(weights, filename):
    with open(filename, 'w') as file:
        # Write the bias term followed by the remaining weights
        file.write(" ".join(map(str, [weights[-1]] + list(weights[:-1]))))



if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        sys.exit(1)

    # Load training data from three files
    train_files = sys.argv[1:]
    X_train = np.concatenate([load_data(file)[0] for file in train_files], axis=0)
    y_train = np.concatenate([load_data(file)[1] for file in train_files], axis=0)

    # Create and train the perceptron
    perceptron = Perceptron()
    perceptron.train(X_train, y_train)

    # Save trained weights to corresponding files
    for i, file in enumerate(train_files):
        weights_filename = f"weights{i+1}.txt"
        save_weights(perceptron.weights, weights_filename)
        print(f"Trained weights saved to {weights_filename}")
