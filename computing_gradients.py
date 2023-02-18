import torch
import numpy as np
# f = w * x
# f = 2 * x
X = np.array([1,2,3,4],dtype = np.float32)
Y = np.array([2,4,6,8],dtype = np.float32)

w = 1.0

# model prediction
def forward(x):
  return w * x;

# loss
def loss(y,y_predicted):
  return ((y_predicted - y)**2).mean()

# gradient 
# MSE = 1/N *(w*x - y) ** 2
# dJ/dw = 1 / N * 2 * x(w * x - y)

def gradient(x,y,y_pred):
  return np.dot(2*x,y_pred - y).mean()

print(f'Prection before training : f(5) = {forward(5):.3f}')

# Training

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
  # prediction = forward pass
  y_pred = forward(X)

  # loss 
  l = loss(Y,y_pred)

  # gradient = backward pass 
  # l.backward() dloss/dw
  # with torch.no_grad():
  #     w -= learning_rate*w.grad()
 
  # zero gradient
  w.grad.zero_()
 
  #dw = gradient(X,Y,y_pred)

  
  if epoch % 2 == 0:
    print(f'epoch {epoch + 1}: w = {w:.3f}, loss - {l:.8f}')

print(f'Prection after training : f(5) = {forward(5):.3f}')
