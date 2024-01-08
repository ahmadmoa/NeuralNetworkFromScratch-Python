# Neural Network Implementation in Python

## Overview

This Python project focuses on implementing a neural network from scratch using the NumPy library. The neural network is designed for classification tasks with three classes. Users can customize various parameters such as the number of hidden layers, neurons in each layer, learning rate, activation function, and more.

## Project Components

### 1. Initialization of Weights

The `initializing_Weights` function initializes the weights of the neural network. Users can specify the number of hidden layers, neurons in each layer, and whether to include bias. Weights are randomly initialized.

### 2. Data Encoding

The `encode` function encodes class labels into one-hot vectors, facilitating the training process.

### 3. Training the Neural Network

The `train_model` function is the core of the training process. It performs forward and backward passes, updating weights through backpropagation. Training options include early stopping. The function prints training and testing accuracies, along with a confusion matrix.

### 4. Testing the Model

The `test_model` function evaluates the trained model on a separate test set, providing testing accuracy and a confusion matrix.

## Usage

- Customize the neural network architecture, training parameters, and activation functions.
- The project utilizes pandas for data manipulation and scikit-learn for data splitting during training.

## Output

The project prints detailed information, including training and testing accuracies, the confusion matrix, and weight matrices if requested.

## Dataset Assumption

- The dataset is assumed to have three classes (C1, C2, C3), and the neural network is intended for a classification task.

## Note

This project serves as an educational tool for understanding the implementation details of a neural network and experimenting with various configurations.

