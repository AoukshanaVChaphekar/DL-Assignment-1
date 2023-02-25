from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
import random
import wandb
import argparse

class FeedForward :

  
  def __init__(self):
        
        self.parameters = {
          "wandb_project":"DL Assignment 1",
          "wandb_entity" : "myname",
          "dataset" : "fashion_mnist",
          "epochs" : 1,
          "batch_size" : 4,
          "loss" : "cross_entropy",
          "optimizer" : "gd",
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
          "activation" : "sigmoid",
          "output_function" : "softmax"
        }

        # update paramters as given in cmd 
        self.update_parameters()

        print(self.parameters)
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

        self.L = self.parameters["num_layers"] + 1
        self.k = len(self.title)
        self.nnl = self.parameters["hidden_size"]
        self.d = self.x_train.shape[1] * self.x_train.shape[2]
        self.n = self.train_n_samples

        # initialize wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.parameters["wandb_project"],
            # entity= self.parameters["wandb_entity"]
          )
  
  def update_parameters(self):
        parser = argparse.ArgumentParser(description='Calculate volume of cylinder')
        parser.add_argument('-wp'   ,'--wandb_project',type = str  ,metavar = '', help='wandb project')
        parser.add_argument('-we'   ,'--wandb_entity' ,type = str  ,metavar = '', help='wandb entity')
        parser.add_argument('-d'    ,'--dataset'      ,type = str  ,metavar = '', help='dataset')
        parser.add_argument('-e'    ,'--epochs'       ,type = int  ,metavar = '', help='epochs')
        parser.add_argument('-b'    ,'--batch_size'   ,type = int  ,metavar = '', help='batch size')
        parser.add_argument('-l'    ,'--loss'         ,type = str  ,metavar = '', help='loss')
        parser.add_argument('-o'    ,'--optimizer'    ,type = str  ,metavar = '', help='optimizer')
        parser.add_argument('-lr'   ,'--learning_rate',type = float,metavar = '', help='learning rate')
        parser.add_argument('-m'    ,'--momentum'     ,type = float,metavar = '', help='momentum')
        parser.add_argument('-beta' ,'--beta'         ,type = float,metavar = '', help='beta')
        parser.add_argument('-beta1','--beta1'        ,type = float,metavar = '', help='beta1')
        parser.add_argument('-beta2','--beta2'        ,type = float,metavar = '', help='beta2')
        parser.add_argument('-eps'  ,'--epsilon'      ,type = float,metavar = '', help='epsilon')
        parser.add_argument('-w_d'  ,'--weight_decay' ,type = float,metavar = '', help='weight decay')
        parser.add_argument('-w_i'  ,'--weight_init'  ,type = str  ,metavar = '', help='weight init')
        parser.add_argument('-nhl'  ,'--num_layers'   ,type = int  ,metavar = '', help='num layers')
        parser.add_argument('-sz'   ,'--hidden_size'  ,type = int  ,metavar = '', help='hidden size')
        parser.add_argument('-a'    ,'--activation'   ,type = str  ,metavar = '', help='activation')
        parser.add_argument('-of'    ,'--output_function'   ,type = str  ,metavar = '', help='output function')
        args = parser.parse_args()
        
        if(args.wandb_project != None):
              self.parameters["wandb_project"] = args.wandb_project
        if(args.wandb_entity != None):
              self.parameters["wandb_entity"] = args.wandb_entity
        if(args.dataset != None):
              self.parameters["dataset"] = args.dataset
        if(args.epochs != None):
              self.parameters["epochs"] = args.epochs
        if(args.batch_size != None):
              self.parameters["batch_size"] = args.batch_size
        if(args.loss != None):
              self.parameters["loss"] = args.loss
        if(args.optimizer != None):
              self.parameters["optimizer"] = args.optimizer
        if(args.learning_rate != None):
              self.parameters["learning_rate"] = args.learning_rate
        if(args.momentum != None):
              self.parameters["momentum"] = args.momentum
        if(args.beta != None):
              self.parameters["beta"] = args.beta
        if(args.beta1 != None):
              self.parameters["beta1"] = args.beta1
        if(args.beta2 != None):
              self.parameters["beta2"] = args.beta2
        if(args.epsilon != None):
              self.parameters["epsilon"] = args.epsilon
        if(args.weight_decay != None):
              self.parameters["weight_decay"] = args.weight_decay
        if(args.weight_init != None):
              self.parameters["weight_init"] = args.weight_init
        if(args.num_layers != None):
              self.parameters["num_layers"] = args.num_layers
        if(args.hidden_size != None):
              self.parameters["hidden_size"] = args.hidden_size
        if(args.activation != None):
              self.parameters["activation"] = args.activation

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

  # function used to implement different activation functions
  def activation_func(self,x,function_name):
        if function_name == "sigmoid":
              return 1 / (1 + np.exp(-x))

  # function used to implement different output functions
  def output_func(self,x,function_name):
    if function_name == "softmax":
      max_element = x.max()
      x = x / max_element
      return np.exp(x) / sum(np.exp(x))  

  # function generating one-hot vector 
  def oneHotVector(self,size,index):
    oneHot = np.zeros(size)
    oneHot[index] = 1
    return oneHot

  # function returning the gradient of function given as an input
  def differentiation(self,function_name,x):
    if function_name == "sigmoid":
      return self.activation_func(x,"sigmoid")*(1 - self.activation_func(x,"sigmoid"))

  # function returning the loss function value
  def loss_function(self,y_predicted,function_name,index):
    if function_name == "cross_entropy":
      return -np.log(y_predicted[index])

  # forward propagation - returns pre_activation vector,post_activation vector and predicted_y vector for each input
  def forward_propagation(self,input,index,weights,bias):
        
    # input = self.x_train
    # dictionary storing pre_activation vectors from each layer
    pre_activation = {}

    # dictionary storing post_activation vectors from each layer
    post_activation = {}
    
    L = self.L
    # Populating pre_activation and post_activation vectors to dictionary in each layer for input[index]
    for k in range(1,L):

      # for first layer,post activation will be input
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
      pre_activation["a" + str(k)] = bias["b" + str(k)] + np.dot(weights["w" + str(k)],np.reshape(post_activation["h" + str(k - 1)],(-1,1)))
      
      # computing h(k) = g(a(k)) where g is activation function
      post_activation["h" + str(k)] = self.activation_func(pre_activation["a" + str(k)],self.parameters["activation"])
      
    # computing pre_activation for last layer
    pre_activation["a"+ str(L)] = bias["b" + str(L)] + np.dot(weights["w" + str(L)],post_activation["h" + str(L - 1)])

    # prediction y (y_hat) = O(a(L)) where O is output function
    predicted_y = self.output_func(pre_activation["a" + str(L)],self.parameters["output_function"])
    
    return pre_activation,post_activation,predicted_y

    # performs back propagation and returns gradients of weights and bias
  
  def backward_propagation(self,index,weights,bias,pre_activation,post_activation,predicted_y,actual_y):
    grad_pre_activation = {}
    grad_post_activation ={}
    grad_weights = {}
    grad_bias = {}
    L = self.L
    # Computing output gradient
    one_hot_vector = np.reshape(np.exp(self.oneHotVector(self.no_of_label,self.y_train[index])),(-1,1))
    grad_pre_activation["a" + str(L)] = (predicted_y - one_hot_vector)
    k = L

    while k > 0:

      # Computing gradient w.r.t parameters - weight and bais
      '''
        np.reshape(grad_pre_activation["a" + str(L)],(-1,1)) = (k,1)
        np.reshape(post_activation["h" + str(L - 1)],(1,-1)) = (1,nnl)
      '''
      grad_weights["w" + str(k)] = np.dot(grad_pre_activation["a" + str(k)],np.reshape(post_activation["h" + str(k - 1)],(1,-1)))
      grad_bias["b" + str(k)] = grad_pre_activation["a" + str(k)]

      if k != 1:
        # Computing gradient differentiationt w.r.t layer below (post_activation)
        grad_post_activation["h" + str(k - 1)] = np.dot(weights["w" + str(k)].T,grad_pre_activation["a" + str(k)])

        # Computing gradient w.r.t layer below (pre_activation)
        g_dash = self.differentiation(self.parameters["activation"],pre_activation["a" + str(k - 1)])
        grad_pre_activation["a" + str(k - 1)] = np.multiply(grad_post_activation["h" + str(k - 1)],g_dash)

      k = k - 1
    return grad_weights,grad_bias

  def make_accumalate_zero(self):

        nnl = self.nnl
        d = self.d
        k = self.k
        L = self.L
        
        acc_grad_weights = {}
        acc_grad_bias = {}
        
        acc_grad_weights["w1"] = np.zeros((nnl,d))
        for i in range(2,L):
          acc_grad_weights["w" + str(i)] = np.zeros((nnl,nnl))
        acc_grad_weights["w" + str(L)] = np.zeros((k,nnl))

        for i in range(1,L):
          acc_grad_bias["b" + str(i)] = np.reshape(np.zeros((nnl)),(-1,1))
        acc_grad_bias["b" + str(L)] = np.reshape(np.zeros((k)),(-1,1))
        return acc_grad_weights,acc_grad_bias
           
  # nnl = number of neurons in each layer,hl = number of hidden layers
  def feed_forward(self):
    nnl = self.nnl
    d = self.d
    k = self.k
    L = self.L
        
    n = self.n

    # d = dimension of input
    
    # parameters - weights,bias
    weights = {}
    bias = {}

    input = self.x_train
    actual_y = self.y_train

    # number of iterations
    # TODo what is use of batch size 
    max_iter = 50
    epoch = self.parameters["epochs"]
    
    
    # step size
    # TODO batch size ?
    # step_size = epoch + 1

    # initailzation of weights
    '''
        W1 = (d,nnl)
        W2,..,W(L - 1) = (nnl,nnl)
        WL = (k,nnl)
    '''
    w1 = np.random.rand(nnl,d)
    weights["w1"] = w1
    for i in range(2,L):
      weights["w" + str(i)] = np.random.rand(nnl,nnl)
    weights["w" + str(L)] = np.random.rand(k,nnl)

    
    # initialization of bias
    for i in range(1,L):
      bias["b" + str(i)] = np.reshape(np.random.rand(nnl),(-1,1))
    bias["b" + str(L)] = np.reshape(np.random.rand(k),(-1,1))

    loss = []
      
    while (epoch < max_iter):
      loss_input = 0

      # to accumulate grad_weights and grad_bais for each epoch
      acc_grad_weights,acc_grad_bias = self.make_accumalate_zero()

      for index in range(n):
        
        # forward propagation
        pre_activation,post_activation,predicted_y = self.forward_propagation(input,index,weights,bias)
        # compute loss
        loss_input += self.loss_function(predicted_y,self.parameters["loss"],self.y_train[index])

        # backward propagation
        grad_weights,grad_bias = self.backward_propagation(index,weights,bias,pre_activation,post_activation,predicted_y,actual_y)

        # accumulate grad_weights and grad_bais for each input
        
        for (key,value) in grad_weights.items():
          acc_grad_weights[key] = acc_grad_weights[key] + grad_weights[key]

        for (key,value) in grad_bias.items():
          acc_grad_bias[key] = acc_grad_bias[key] + grad_bias[key]
        
        acc_grad_weights,acc_grad_bias,weights,bias = self.optimization_function(function_name = self.parameters["optimizer"],
                                                                                  index = index,
                                                                                  acc_grad_weights = acc_grad_weights,
                                                                                  acc_grad_bias= acc_grad_bias,weights = weights,
                                                                                  bias = bias)
      epoch = epoch + 1
      loss.append(loss_input/n)

    index = np.random.randint(self.test_n_samples)
    input = self.x_test
    pre_activation,post_activation,predicted_y = self.forward_propagation(input,index,weights,bias)

    print(predicted_y*100)

  
  def optimization_function(self,function_name,index,acc_grad_weights,acc_grad_bias,weights,bias):
        nnl = self.nnl
        d = self.d
        k = self.k
        L = self.L
        step_size = self.parameters["learning_rate"]

        if (function_name == "gd" and index == (self.n - 1)) or function_name == "sgd":
              for (key,value) in weights.items():
                weights[key] = np.subtract(weights[key],step_size * acc_grad_weights[key])
              for (key,value) in bias.items():
                bias[key] = np.subtract(bias[key],step_size * acc_grad_bias[key])
              self.make_accumalate_zero()
        return acc_grad_weights,acc_grad_bias,weights,bias


feed_forward = FeedForward()
# feed_forward.question_1()
feed_forward.feed_forward()
