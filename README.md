# Fashion-MNIST Classification using Feed forward neural networks

This folder contains the code base for Assignment 1

The wandb report can be found in the following link:
https://wandb.ai/cs22m019/DL%20Final%20Assignment%201/reports/DL-Report--VmlldzozNzM5NDYx

The problem statement involves implementing a feedforward neural network and backpropagation code for training the network on Fashion-MNIST dataset

The code base has the following features:

    1.Forward and backward propagation algorithms hardcoded.
    2.The weights and biases, activations and their gradients are stored as dictionaries configured as attributed within the FeedForward class.
    3.A class is created which has functions to implement all the required output_functions,activation_functions,loss_functions,optimizers,
    forward and backward propagation algorithms.
    4.When the class is instantiated,the default hyperparamteres are set namely, the number of layers, hidden neurons, activation function,
    optimizer, weight decay,etc.
    
# Code structure

These are 4 files uploaded to github.

  1.train.py - this file contains the entire code for feedforward neural network with default hyperparameters set to the configuration 
             where the best accuracy was obtained.

Below are the 3 Google colab note books which contain FNN code from train.py to carry out training and hyperparameter search using Wandb for various
hyper parameter combinations.
  2.DL_Assignment_1_bayes.ipynb - this notebook generates sweeps using the Bayesian method for cross_entropy loss.
  3.DL_Assignment_1_random_MSE.ipynb - this notebook generates sweeps using the Random search method for mean-squared error loss.
  4.DL_Assignment_1_random_cross_entropy.ipynb - this notebook generates sweeps using the Random search method for cross-entropy loss.


# Running files
