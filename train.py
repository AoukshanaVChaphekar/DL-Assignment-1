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

# list of label titles - actual output
title = ["T-shirt/top","Trouser","PullOver","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

def question_1():
    # dictionary of labels added 
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

# question_1()

# forward propagation - returns pre_activation vector,post_activation vector and predicted_y vector for each input
def forward_propagation(weights,input,bias,L,index):
  
  # dictionary storing pre_activation vectors from each layer
  pre_activation = {}

  # dictionary storing post_activation vectors from each layer
  post_activation = {}

  # Populating pre_activation and post_activation vectors to dictionary in each layer for input[index]
  for k in range(1,L):

    # for first layer,post activation will be inputs
    if(k == 1):
      ''' flattening the input: 
          -input(60000,28,28)
          -input[index] size = (28,28)
          -flattening input[index] gives size (784,1) = (d,1) where d is dimension of input
          post_activation[h0] size = (d,1)
          bias[b1] size = (nnl,1)
          weights[w1] size = (nnl,d)
          Therefore we get pre_activation[a1] size = (nnl,1) for all layer except last layer
      '''
      post_activation["h" + str(k - 1)] = input[index].flatten()

    # computing a(k) = b(k) + w(k)*h(k - 1) for each input[index]
    pre_activation["a" + str(k)] = bias["b" + str(k)] + np.matmul(weights["w" + str(k)],post_activation["h" + str(k - 1)])
    
    # computing h(k) = g(a(k)) where g is activation function for each input[index]
    post_activation["h" + str(k)] = activation_func(pre_activation["a" + str(k)],"sigmoid")
    
    # computing pre_activation for last layer for each input[index]
    pre_activation["a"+ str(L)] = bias["b" + str(L)] + np.matmul(weights["w" + str(L)],post_activation["h" + str(L - 1)])

    # prediction y (y_hat) = O(a(L)) where O is output function
    predicted_y = output_func(pre_activation["a" + str(L)],"softmax")

  return pre_activation,post_activation,predicted_y

