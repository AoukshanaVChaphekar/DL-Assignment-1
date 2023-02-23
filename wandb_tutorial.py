import wandb
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
# def load_image():

def question_1():

  # dictionary of labels to be added 
  wandb.init(
    # set the wandb project where this run will be logged
    project="wandb tutorial",
  )
  features = x_train
  labels = y_train
  labels_added = {}
  

  ''' 
      Running the loop for the number of training samples.
      In each iteration,a random index is generated and we extract the feature and label at the generated index.
      If the label is already in the labels_added dictionary,we ignore that label,else we add that (label,feature) 
      as (key,value) pair in dictionary (so that one label is considered only once).
  '''
  examples = []
  for i in range(train_n_samples):
    index = random.randrange(train_n_samples)
    feature = x_train[index]
    label = y_train[index]
    if(label in labels_added.keys()):
      continue
    labels_added[label] = feature
    image = wandb.Image(labels_added[label], caption=f"{title[label]}")
    examples.append(image)
  wandb.log({"images": examples})

question_1()

