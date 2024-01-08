# Neural Network Implementation in Python

## Overview

This Python project focuses on building a neural network from scratch using the NumPy library. The neural network is designed for classification tasks with three classes, and users can customize various parameters such as the number of hidden layers, neurons in each layer, learning rate, activation function, and more.

## Project Components

### 1. Initialization of Weights

The `initializing_Weights` function initializes the weights of the neural network. Users can specify the number of hidden layers, neurons in each layer, and whether to include bias. Weights are randomly initialized for effective training.

### 2. Data Encoding

The `encode` function facilitates the training process by converting class labels into one-hot vectors. This encoding scheme enhances the network's ability to learn and generalize from the provided data.

### 3. Training the Neural Network

The `train_model` function is the core of the training process. It performs forward and backward passes, updating weights through backpropagation. Training options include early stopping to prevent overfitting. The function provides detailed information, including training and testing accuracies, along with a confusion matrix.

### 4. Testing the Model

The `test_model` function evaluates the trained model on a separate test set, providing testing accuracy and a confusion matrix for assessing the model's performance.

## Usage

- Customize the neural network architecture, training parameters, and activation functions to suit your specific task.
- The project leverages pandas for data manipulation and scikit-learn for data splitting during training, ensuring a seamless integration into your workflow.

## Output

The project prints comprehensive information, including training and testing accuracies, the confusion matrix, and, if requested, weight matrices for transparency in model analysis.

## Data Preprocessing

Before training the neural network, it's essential to preprocess the dataset. The 'Dry_Bean_Dataset.csv' undergoes standard preprocessing steps, including scaling numerical features and encoding categorical data. These steps ensure that the data is appropriately prepared for training the neural network.

