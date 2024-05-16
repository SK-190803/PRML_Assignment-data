import sys
import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None

    def load_weights(self, weights_filename):
        with open(weights_filename, 'r') as file:
            self.weights = np.array(list(map(float, file.readline().split())))

    def predict(self, X):
        X_with_bias = np.c_[X, np.ones(X.shape[0])]  # Add bias term
        return np.where(np.dot(X_with_bias, self.weights) >= 0, 1, 0)

def load_data(filename, has_labels=True):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_samples = int(lines[0])
        num_features = 4 if has_labels else 5
        start_line = 1 if has_labels else 0
        data = np.zeros((num_samples, num_features))
        labels = np.zeros(num_samples, dtype=int) if has_labels else None
        for i, line in enumerate(lines[start_line:]):
            values = line.strip().split()
            data[i] = list(map(float, values[:num_features]))  # Load as floats
            if has_labels:
                labels[i] = int(values[-1])
    if has_labels:
        return data, labels
    else:
        return data

def load_weights(weights_filename):
    with open(weights_filename, 'r') as file:
        weights = np.array(list(map(float, file.readline().split())))
    return weights

def evaluate_model(y_pred, y_true):
    num_correct = np.sum(y_pred == y_true)
    accuracy = num_correct / len(y_true)
    return accuracy

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 3:
        print("Usage: python test.py test.txt weights1.txt weights2.txt weights3.txt")
        sys.exit(1)

    # Load test data from file
    test_file = sys.argv[1]
    X_test, y_test_true = load_data(test_file, has_labels=True)

    # Load weights from files
    num_weights = len(sys.argv) - 2
    weights_files = sys.argv[2:]
    accuracies = []

    for i in range(num_weights):
        # Load weights from file
        weights = load_weights(weights_files[i])

        # Create and load the perceptron model
        perceptron = Perceptron()
        perceptron.weights = weights

        # Calculate labels using the perceptron model
        y_test_predictions = perceptron.predict(X_test)

        # Compute accuracy for the perceptron
        accuracy = evaluate_model(y_test_predictions, y_test_true)
        accuracies.append(accuracy)

        print(f"Accuracy for weights{i+1}: {accuracy}")

