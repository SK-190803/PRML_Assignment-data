Approach or Thought Process for Question 1-
------------------------------------------------------------------------------------------------------------------------------------------------------------
Description of DataSet Generation-

This Python script generates a synthetic dataset and performs data splitting for machine learning experiments. The process involves the following steps:

Synthetic Data Generation:
The script generates a synthetic dataset comprising random input vectors (X) and corresponding labels (y) using a perceptron model.
The perceptron model's weights are randomly generated for the dataset.

Data Splitting:
The generated dataset is split into training and testing sets with a 70:30 ratio.
The training set comprises 70% of the total samples, while the testing set contains the remaining 30%.

Saving Data to Files:
Both the training and testing sets are saved into separate text files.
The training data is saved in a file named "train.txt", while the testing data is saved in a file named "test.txt".

File Description
data.txt: Contains the generated synthetic dataset along with labels.
train.txt: Contains the training set data and labels.
test.txt: Contains the testing set data and no labels
---------------------------------------------------------------------------------------------------------------------------------------------------------------

Description of train.py file-

This Python script trains a Perceptron model on a given dataset and saves the trained weights to a file for later use. The script performs the following tasks:

Perceptron Class:
The script defines a Perceptron class with methods to train the model and save the trained weights.
The perceptron is trained using the perceptron learning algorithm, adjusting weights based on prediction errors.

Data Loading:
The script provides a function load_data(filename) to load training data from a file.
The data file should contain input features and corresponding labels.

Training:
The script loads the training data from a file specified via command-line arguments.
The Perceptron model is instantiated, trained on the loaded data, and weights are updated accordingly.

Saving Weights:
After training, the script saves the trained weights to a file named "weights.txt".

File Description
train.txt: Contains the training data, including input features and corresponding labels.
weights.txt: Stores the trained weights of the Perceptron model.

In order to run train.py file ,use following command

python B22ES006_train.py B22ES006_train.txt
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Description of test.py file-

This Python script loads test data and pre-trained weights of a Perceptron model to evaluate its performance. The script performs the following tasks:

Perceptron Class:
The script defines a Perceptron class with methods to load pre-trained weights and make predictions.

Data Loading:
The script provides a function load_data(filename) to load test data from a file.
The test data file should contain input features

Evaluation:
The script loads test data and pre-trained weights from files specified via command-line arguments.
The pre-trained weights are loaded into the Perceptron model.
The model makes predictions on the test data using the loaded weights and print the generated labels.
The accuracy of the model is computed by comparing the predicted labels with the true labels.

Output:
The script prints the predicted labels, true labels, and the computed accuracy of the model.

File Description
test.txt: Contains the test data, including input features only
weights.txt: Stores the pre-trained weights of the Perceptron model.
test.py: Python script for evaluating the Perceptron model using test data and pre-trained weights.

in order to run the test.py file --use following command-
python B22ES006_test.py B22ES006_test.txt weights.txt

note-weights.txt file will be generated after execution of train.py file

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------