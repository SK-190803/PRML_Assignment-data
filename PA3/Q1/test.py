import sys
import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None

    def load_weights(self, weights_filename):
        with open(weights_filename, 'r') as file:
            weights = list(map(float, file.readline().split()))
            self.weights = np.array(weights)

    def predict(self, X):
        return np.where(np.dot(X, self.weights[:-1]) + self.weights[-1] >= 0, 1, 0)

def load_data(filename, has_labels=True):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_samples = len(lines)
        num_features = 4 if has_labels else 5
        data = np.zeros((num_samples, num_features))
        labels = np.zeros(num_samples, dtype=int) if has_labels else None
        for i, line in enumerate(lines):
            values = line.strip().split()
            data[i] = list(map(float, values[:num_features]))  # Load as floats
            if has_labels:
                labels[i] = int(values[-1])
    if has_labels:
        return data, labels
    else:
        return data

def evaluate_model(y_pred, y_true):
    num_correct = np.sum(y_pred == y_true)
    accuracy = num_correct / len(y_true)
    return accuracy

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 7:
        print("Usage: python test.py test_m20.txt test_m50.txt test_m70.txt weights1.txt weights2.txt weights3.txt")
        sys.exit(1)

    # Load test data from three files
    test_files = sys.argv[1:4]
    X_test = [load_data(file, has_labels=True)[0] for file in test_files]
    y_test_true = [load_data(file, has_labels=True)[1] for file in test_files]

    # Load weights from three files
    weight_files = sys.argv[4:]
    weights = [load_data(file, has_labels=False)[0] for file in weight_files]

    # Calculate labels using each perceptron model
    y_test_predictions = []
    for i in range(3):
        perceptron = Perceptron()
        perceptron.weights = weights[i]
        y_test_predictions.append(np.array([perceptron.predict(x) for x in X_test[i]]))

    # Compute accuracy for each perceptron
    accuracies = []
    for i in range(3):
        accuracy = evaluate_model(y_test_predictions[i], y_test_true[i])
        accuracies.append(accuracy)
        print(f"Accuracy for perceptron {i+1}: {accuracy}")
print("Perceptron 1 refers to train and test on 20% of data")
print("Perceptron 2 refers to train and test on 50% of data")
print("Perceptron 3 refers to train and test on 70% of data")
