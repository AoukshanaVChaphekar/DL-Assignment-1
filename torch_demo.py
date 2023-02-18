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



# if we want to compute graident on tensor later
x = torch.ones(5,requires_grad = True)
print(x)

# requires_grad - if we want to find gradient of any func wrt x later
x = torch.randn(3,requires_grad = True)
print(x) 
# a computational graph is created whenver we perform some operation on x
y = x + 2
print(y)
y = y.mean() # scalar
y.backward() # dy/dx

print(x.grad)

z = x + 2 # vector
v = torch.tensor([0.1,1.0,0.001],dtype = torch.float32)
z.backward(v)
print(x.grad) # gradient

# removing a gradient
# x.detach(),x.requires_grad_(false),
# with torch.no_grad(): 
#      y = x + 2 removes gradient

# empty graident x.grad.zero_()

# backpropagation
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad = True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

# backward pass
loss.backward() # d(loss)/ dw
print(w.grad)

## update weights and 
## do next forward,backward pass for several iteraions to minimize loss 