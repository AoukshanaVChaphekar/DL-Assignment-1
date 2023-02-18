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


# converting tensor to numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# if tensor is on cpu then if we modify a,b gets modified
# converting numpy to tensor
a = np.ones(5)
print(a)

b = torch.from_numpy(a)
print(b)

# if numpy is modified,tensor is also modified .
# this happens if tensor is on Cpu  
