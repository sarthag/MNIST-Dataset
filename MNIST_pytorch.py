#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


# In[2]:


#define the NN model
class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)
    self.fc2 = nn.Linear(50, num_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# In[3]:


#basic test on the model
model = NN(784 ,10)
x = torch.randn(64, 784)
print(model(x).shape)


# In[4]:


#parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_ephocs = 100


# In[5]:


#loading the data

X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = X.astype(np.float32)
y = np.int_(y)


# In[6]:


X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
y_tensor = y_tensor.type(torch.LongTensor)
X_train, X_test, y_train, y_test = train_test_split(X_tensor,y_tensor, test_size = (1/7), random_state = 42)


# In[7]:


#initialise network
model = NN(input_size, num_classes)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate) 


# In[8]:


#Train the network
for epoch in range(num_ephocs):
    batch = 0
    while batch < len(y_train) - 64:
        batch_next = batch+64
        data = X_train[batch: batch_next]
        targets = y_train[batch: batch_next]
        batch = batch_next

        
        #Forward Prop
        scores = model(data)
        loss = loss_fun(scores, targets)
        
        #Back prop
        optimizer.zero_grad()
        loss.backward()
        
        #Optimizer
        optimizer.step()


# In[9]:


def chk_accuracy(X_inp, y_inp, model):
    
    batch = 0
    num_correct = 0
    num_samples = len(y_inp) - len(y_inp)%64
    with torch.no_grad():
        while batch < len(y_inp) - 64:
            batch_next = batch+64
            x = X_inp[batch: batch_next]
            y = y_inp[batch: batch_next]
            batch = batch_next

            scores = model(x)
            predictions = scores.argmax(1)
            num_correct += sum((predictions == y))
            
        return float(num_correct)/float(num_samples)


# In[10]:


print("Train Accuracy:", chk_accuracy(X_train, y_train, model))
print("Test Accuracy:", chk_accuracy(X_test, y_test, model))

