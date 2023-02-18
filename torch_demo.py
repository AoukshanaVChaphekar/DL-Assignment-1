import torch
import numpy as np

# creating empty tensor with size 1 - values are random
x = torch.empty(1,2)
print(x)

# random value tensor of szie 2*3
x = torch.rand(2,3)
print(x)

# giving custom datatype
x = torch.zeros(1,2,dtype = torch.int)
print(x)
x = torch.ones(1,2)
print(x)

# creating tensor from python list
x = torch.tensor([2,3,4])
print(x)

# reshaping
x = torch.rand(4,4)
print(x)

y = x.view(16)
print(y)

# 2 * 8 
y = x.view(-1,8)
print(y.size())
