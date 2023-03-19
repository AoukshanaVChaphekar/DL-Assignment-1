# Fashion-MNIST Classification using Feed Forward Neural Networks

The **wandb report** can be found in the following link:

https://wandb.ai/cs22m019/DL%20Final%20Assignment%201/reports/DL-Report--VmlldzozNzM5NDYx

The problem statement involves implementing a feedforward neural network and backpropagation code for training the network on Fashion-MNIST dataset

## The code base has the following features:

    1.Forward and backward propagation algorithms are hardcoded.
    
    2.The weights and biases, activations and their gradients are stored as dictionaries configured as attributes within the FeedForward class.
    
    3.A class is created which has functions to implement all the required output_functions,activation_functions,loss_functions,optimizers,
    forward and backward propagation algorithms.
    
    4.When the class is instantiated,the default hyperparamteres are set namely, the number of layers, hidden neurons, activation function,
    optimizer, weight decay,etc.
    
# Code structure

## These are 4 files uploaded to github:

    train.py - this file contains the entire code for feedforward neural network with default hyperparameters set to the configuration 
               where the best accuracy was obtained.

Below are the 3 Google colab note books which contain FNN code from train.py to carry out training and hyperparameter search using Wandb for various
hyper parameter combinations:

    1.DL_Assignment_1_bayes.ipynb - this notebook generates sweeps using the Bayesian method for cross_entropy loss.
    
    2.DL_Assignment_1_random_MSE.ipynb - this notebook generates sweeps using the Random search method for mean-squared error loss.
    
    3.DL_Assignment_1_random_cross_entropy.ipynb - this notebook generates sweeps using the Random search method for cross-entropy loss.


# Running the files

## Quesiton 1:
Uncomment the line number 1431 in train.py

Execute the command
     
```python
python train.py
```  

Generated output will be visible as a run in Wandb workspace.
    
## Question 2:
Uncomment the line number 1434 in train.py 
    
Execute the command
```python 
python train.py
```
    
The output will be generated in terminal window.
    
## Quesiton 7:
Uncomment the line number 1438 or 1439 in train.py to generate confusion matrix for train dataset and test dataset respectively.

Execute the command
```python 
python train.py
```

## Quesiton 10:
Uncomment the line number 1442 to 1490 in train.py to obtain the accuracies for the following 3 configurations.

Execute the command
```python 
python train.py
```
## Configuration 1
    
    Number of Hidden Layers - 4
    Number of Hidden Neurons - 128
    Weight Decay - 0.5
    Activation - tanh
    Initialisation - Xavier 
    Optimiser - NADAM
    Learning Rate - 0.001
    Batch size - 32
Accuracy obtained - 86.8%
## Configuration 2
    
    Number of Hidden Layers - 5
    Number of Hidden Neurons - 64
    Weight Decay - 0.0005
    Activation - tanh
    Initialisation - Xavier
    Optimiser - NADAM
    Learning Rate - 0.001
    Batch size - 32
Accuracy obtained - 86.48%
## Configuration 3
    
    Number of Hidden Layers - 4
    Number of Hidden Neurons - 64
    Weight Decay - 0.5
    Activation - tanh
    Initialisation - Xavier
    Optimiser - RMSProp
    Learning Rate - 0.001
    Batch size - 32
Accuracy obtained - 86.9%

# Results

For the feed forward neural network implemented, the maximum validation accuracy reported was 86.62% on the Fashion MNIST dataset.
One of the model configuration chosen to be the best is as follows:

    Number of Hidden Layers - 4
    Number of Hidden Neurons - 128
    Weight Decay - 0.5
    Activation - tanh
    Initialisation - Xavier 
    Optimiser - NADAM
    Learning Rate - 0.001
    Batch size - 32

    
