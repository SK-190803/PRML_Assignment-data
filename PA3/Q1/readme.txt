Approach or Thought Process for Question 1-

Description of train.py file
This Python script generates synthetic data consisting of input vectors and corresponding labels based on randomly generated weights. It then splits this data into subsets of 20%, 50%
,and 70%, and further splits each subset into training and testing sets.After that, it defines a Perceptron class with methods for training and predicting using the perceptron model. 
The train method uses gradient descent to update the weights based on the training data.The script also includes functions for loading and saving data, as well as for normalizing data
and saving weights to files.In the main part of the script, it loads the training data from command-line arguments, concatenates them, trains a perceptron model using this data, and 
saves the trained weights to corresponding files.

Description of train.py file
This Python script evaluates the performance of three perceptron models on test data loaded from separate files. It loads test data and weights for each perceptron from command-line
arguments. It then calculates predictions using each perceptron model and evaluates their accuracy against true labels. Finally, it prints the accuracies of each perceptron model
and provides a description indicating which subset of data each perceptron was trained and tested on.