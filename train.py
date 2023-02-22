from tensorflow import keras
from keras.datasets import fashion_mnist
import numpy as np
from matplotlib import pyplot as plt
import random

# loading training and test data frm fashion_mnist dataset
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# computing number of samples in training and test data
train_n_samples = x_train.shape[0]
test_n_samples = x_test.shape[0]

# list of label titles -> actual output
title = ["T-shirt/top","Trouser","PullOver","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
no_of_label = len(title)

def question_1():
      # dictionary of labels to be added 
  labels_added = {}

  features = x_train
  labels = y_test

  ''' 
      Running the loop for the number of training samples.
      In each iteration,a random index is generated and we extract the feature and label at the generated index.
      If the label is already in the labels_added dictionary,we ignore that label,else we add that (label,feature) 
      as (key,value) pair in dictionary (so that one label is considered only once).
  '''
  for i in range(train_n_samples):
    index = random.randrange(train_n_samples)
    feature = x_train[index]
    label = y_train[index]
    if(label in labels_added.keys()):
      continue
    labels_added[label] = feature

  # Plotting the images
  fig = plt.figure(figsize =(8,8)) 
  fig.suptitle("Images")
  columns = 5
  rows = 2
  for i in range(1, columns*rows +1):
      img = labels_added[i - 1]
      ax1 = fig.add_subplot(rows, columns, i)
      ax1.title.set_text(title[i - 1])
      plt.imshow(img,cmap = "gray")

  # Displaying the plot
  plt.show()

# function used to implement different activation functions
def activation_func(x,function_name):
  if function_name == "sigmoid":
    return 1 / (1 + np.exp(-x))

# function used to implement different output functions
def output_func(x,function_name):
  if function_name == "softmax":
    max_element = x.max()
    x = x / max_element
    return np.exp(x) / sum(np.exp(x))  

# function generating one-hot vector 
def oneHotVector(size,index):
  oneHot = np.zeros(size)
  oneHot[index] = 1
  return oneHot

# function returning the gradient of function given as an input
def differentiation(function_name,x):
  if function_name == "sigmoid":
    return activation_func(x,"sigmoid")*(1 - activation_func(x,"sigmoid"))

# function returning the loss function value
def loss_function(y_predicted,function_name,index):
  if function_name == "cross_entropy":
    return -np.log(y_predicted[index])

# question_1()

