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


class CNN(nn.Module):
    def __init__(self, in_channel =1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size= (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return(x)


# In[3]:


#Basic Test
model = CNN()
x = torch.randn(64,1,28,28)
print(model(x).shape)


# In[4]:


#parameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_ephocs = 10


# In[5]:


#loading the data

X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = X.astype(np.float32)
y = np.int_(y)
X = X.reshape(X.shape[0], 1, 28, 28)
print(X.shape, y.shape)


# In[6]:


X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
y_tensor = y_tensor.type(torch.LongTensor)
X_train, X_test, y_train, y_test = train_test_split(X_tensor,y_tensor, test_size = (1/7), random_state = 42)


# In[7]:


#initialise network
model = CNN()
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

