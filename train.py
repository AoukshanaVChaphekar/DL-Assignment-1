from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
import random
import wandb
class FeedForward :

  
  def __init__(self):
        self.parameters = {
          "wandb_project":"myprojectname",
          "wandb_entity" : "myname",
          "dataset" : "fashion_mnist",
          "epochs" : 1,
          "batch_size" : 4,
          "loss" : "cross_entropy",
          "optimizer" : "sgd",
          "learning_rate" : 0.1,
          "momentum" : 0.5,
          "beta" : 0.5,
          "beta1" : 0.5,
          "beta2" : 0.5,
          "epsilon" : 0.000001,
          "weight_decay" : 0,
          "weight_init" : "random",
          "num_layers" : 1,
          "hidden_size" : 4,
          "activation" : "sigmoid"
        }
        # loading training and test data from fashion_mnist dataset
        if(self.parameters["dataset"] == "fashion_mnist"):
              (self.x_train,self.y_train),(self.x_test,self.y_test) = fashion_mnist.load_data()
        else:
              (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()
        
              

        # computing number of samples in training and test data
        self.train_n_samples = self.x_train.shape[0]
        self.test_n_samples = self.x_test.shape[0]

        # list of label titles -> actual output
        self.title = ["T-shirt/top","Trouser","PullOver","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
        self.no_of_label = len(self.title)

        # initialize wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="DL Assignment 1",
          )
  

  def question_1(self):
      
    features = self.x_train
    labels = self.y_train
    # dictionary of labels to be added 
    labels_added = {}

    ''' 
        Running the loop for the number of training samples.
        In each iteration,a random index is generated and we extract the feature and label at the generated index.
        If the label is already in the labels_added dictionary,we ignore that label,else we add that (label,feature) 
        as (key,value) pair in dictionary (so that one label is considered only once).
    '''
    images = []
    for i in range(self.train_n_samples):
      index = random.randrange(self.train_n_samples)
      feature =self. x_train[index]
      label = self.y_train[index]
      if(label in labels_added.keys()):
        continue
      labels_added[label] = feature
      image = wandb.Image(labels_added[label], caption=f"{self.title[label]}")
      images.append(image)
    wandb.log({"Images": images})

feed_forward = FeedForward()
feed_forward.question_1()
  # # function used to implement different activation functions
  # def activation_func(x,function_name):
  #   if function_name == "sigmoid":
  #     return 1 / (1 + np.exp(-x))

  # # function used to implement different output functions
  # def output_func(x,function_name):
  #   if function_name == "softmax":
  #     max_element = x.max()
  #     x = x / max_element
  #     return np.exp(x) / sum(np.exp(x))  

  # # function generating one-hot vector 
  # def oneHotVector(size,index):
  #   oneHot = np.zeros(size)
  #   oneHot[index] = 1
  #   return oneHot

  # # function returning the gradient of function given as an input
  # def differentiation(self,function_name,x):
  #   if function_name == "sigmoid":
  #     return self.activation_func(x,"sigmoid")*(1 - self.activation_func(x,"sigmoid"))

  # # function returning the loss function value
  # def loss_function(y_predicted,function_name,index):
  #   if function_name == "cross_entropy":
  #     return -np.log(y_predicted[index])

  # # forward propagation - returns pre_activation vector,post_activation vector and predicted_y vector for each input
  # def forward_propagation(self,weights,input,bias,L,index):
        
  #   # dictionary storing pre_activation vectors from each layer
  #   pre_activation = {}

  #   # dictionary storing post_activation vectors from each layer
  #   post_activation = {}

  #   # Populating pre_activation and post_activation vectors to dictionary in each layer for input[index]
  #   for k in range(1,L):

  #     # for first layer,post activation will be input
  #     if(k == 1):
  #       ''' flattening the input: 
  #           -input(60000,28,28)
  #           -input[index] size = (28,28)
  #           -flattening input[index] gives size (784,1) = (d,1) where d is dimension of input
  #           post_activation[h0] size = (d,1)
  #           bias[b1] size = (nnl,1)
  #           weights[w1] size = (nnl,d)
  #           Therefore we get pre_activation[a1] size = (nnl,1) for all layer except last layer
  #       '''
  #       post_activation["h" + str(k - 1)] = input[index].flatten()

  #     # computing a(k) = b(k) + w(k)*h(k - 1) for each input[index]
  #     pre_activation["a" + str(k)] = bias["b" + str(k)] + np.dot(weights["w" + str(k)],np.reshape(post_activation["h" + str(k - 1)],(-1,1)))
      
  #     # computing h(k) = g(a(k)) where g is activation function
  #     post_activation["h" + str(k)] = activation_func(pre_activation["a" + str(k)],"sigmoid")
      
  #   # computing pre_activation for last layer
  #   pre_activation["a"+ str(L)] = bias["b" + str(L)] + np.dot(weights["w" + str(L)],post_activation["h" + str(L - 1)])

  #   # prediction y (y_hat) = O(a(L)) where O is output function
  #   predicted_y = output_func(pre_activation["a" + str(L)],"softmax")
    
  #   return pre_activation,post_activation,predicted_y

  #   # performs back propagation and returns gradients of weights and bias
  # def backward_propagation(index,pre_activation,post_activation,predicted_y,actual_y,L,weights):
  #   grad_pre_activation = {}
  #   grad_post_activation ={}
  #   grad_weights = {}
  #   grad_bias = {}

  #   # Computing output gradient
  #   one_hot_vector = np.reshape(np.exp(oneHotVector(no_of_label,y_train[index])),(-1,1))
  #   grad_pre_activation["a" + str(L)] = (predicted_y - one_hot_vector)
    
  #   k = L

  #   while k > 0:

  #     # Computing gradient w.r.t parameters - weight and bais
  #     '''
  #       np.reshape(grad_pre_activation["a" + str(L)],(-1,1)) = (k,1)
  #       np.reshape(post_activation["h" + str(L - 1)],(1,-1)) = (1,nnl)
  #     '''
  #     grad_weights["w" + str(k)] = np.dot(grad_pre_activation["a" + str(k)],np.reshape(post_activation["h" + str(k - 1)],(1,-1)))
  #     grad_bias["b" + str(k)] = grad_pre_activation["a" + str(k)]

  #     if k != 1:
  #       # Computing gradient differentiationt w.r.t layer below (post_activation)
  #       grad_post_activation["h" + str(k - 1)] = np.dot(weights["w" + str(k)].T,grad_pre_activation["a" + str(k)])

  #       # Computing gradient w.r.t layer below (pre_activation)
  #       g_dash = differentiation("sigmoid",pre_activation["a" + str(k - 1)])
  #       grad_pre_activation["a" + str(k - 1)] = np.multiply(grad_post_activation["h" + str(k - 1)],g_dash)

  #     k = k - 1
  #   return grad_weights,grad_bias
  # def optimization_function(function_name,n,index,weights,bias,acc_grad_weights,acc_grad_bias,step_size,nnl,d,k,L):
  #   if (function_name == "gd" & index == (n - 1)) | function_name == "sgd":
  #         for (key,value) in weights.items():
  #           weights[key] = np.subtract(weights[key],step_size * acc_grad_weights[key])
  #         for (key,value) in bias.items():
  #           bias[key] = np.subtract(bias[key],step_size * acc_grad_bias[key])
  #         make_accumalate_zero(nnl,d,k,L)
  #         return weights,bias
  #   # if (function_name == "momentum_gd" & index == (n - 1)):   
  # def make_accumalate_zero(nnl,d,k,L):
  #     acc_grad_weights = {}
  #     acc_grad_bias = {}
      
  #     acc_grad_weights["w1"] = np.zeros((nnl,d))
  #     for i in range(2,L):
  #       acc_grad_weights["w" + str(i)] = np.zeros((nnl,nnl))
  #     acc_grad_weights["w" + str(L)] = np.zeros((k,nnl))

  #     for i in range(1,L):
  #       acc_grad_bias["b" + str(i)] = np.reshape(np.zeros((nnl)),(-1,1))
  #     acc_grad_bias["b" + str(L)] = np.reshape(np.zeros((k)),(-1,1))
  #     return acc_grad_weights,acc_grad_bias
           
  # # nnl = number of neurons in each layer,hl = number of hidden layers
  # def feed_forward(self):
  #   nnl = self.parameters["num_layers"]
  #   hl = self.parameters["hidden_size"]
  #   optimizer_func = self.parameters["optimizer"] 

  #   n = self.train_n_samples

  #   # d = dimension of input
  #   d = self.x_train.shape[1] * self.x_train.shape[2]

  #   # parameters - weights,bias
  #   weights = {}
  #   bias = {}

  #   input = self.x_train
  #   actual_y = self.y_train

  #   # number of iterations
  #   max_iter = 10
  #   epoch = 0

  #   # k = number of output neurons
  #   k = len(self.title)

  #   # total layers
  #   L = hl + 1

  #   # step size
  #   step_size = epoch + 1

  #   # initailzation of weights
  #   '''
  #       W1 = (d,nnl)
  #       W2,..,W(L - 1) = (nnl,nnl)
  #       WL = (k,nnl)
  #   '''
  #   w1 = np.random.rand(nnl,d)
  #   weights["w1"] = w1
  #   for i in range(2,L):
  #     weights["w" + str(i)] = np.random.rand(nnl,nnl)
  #   weights["w" + str(L)] = np.random.rand(k,nnl)

    
  #   # initialization of bias
  #   for i in range(1,L):
  #     bias["b" + str(i)] = np.reshape(np.random.rand(nnl),(-1,1))
  #   bias["b" + str(L)] = np.reshape(np.random.rand(k),(-1,1))

  #   loss = []
      
  #   while (epoch < max_iter):
  #     loss_input = 0

  #     # to accumulate grad_weights and grad_bais for each epoch
  #     acc_grad_weights,acc_grad_bias = self.make_accumalate_zero(nnl = nnl,d = d,k = k,L = L)

  #     for index in range(n):
        
  #       # forward propagation
  #       pre_activation,post_activation,predicted_y = forward_propagation(weights,input,bias,L,index)
        
  #       # compute loss
  #       loss_input += loss_function(predicted_y,"cross_entropy",y_train[index])

  #       # backward propagation
  #       grad_weights,grad_bias = backward_propagation(index,pre_activation,post_activation,predicted_y,actual_y,L,weights)

  #       # accumulate grad_weights and grad_bais for each input
  #       for (key,value) in grad_weights.items():
  #         acc_grad_weights[key] = acc_grad_weights[key] + grad_weights[key]
  #       for (key,value) in grad_bias.items():
  #         acc_grad_bias[key] = acc_grad_bias[key] + grad_bias[key]
        
  #       acc_grad_weights,acc_grad_bias,weights,bias = optimization_function(function_name= "gd",n = n,index = index,weights = weights,
  #       bias = bias,acc_grad_weights = acc_grad_weights,acC_grad_bias = acc_grad_bias,step_size= step_size,nnl = nnl,d = d,k = k,L = L)
          
  #     epoch = epoch + 1
  #     step_size = epoch
  #     loss.append(loss_input/n)

  #   index = np.random.randint(test_n_samples)
  #   input = x_test
  #   pre_activation,post_activation,predicted_y = forward_propagation(weights,input,bias,L,index)

  #   print(predicted_y*100)
  
