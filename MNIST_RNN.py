# -*- coding: utf-8 -*-
"""MNIST_RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VAR24Iqt89C63XDW6PgnDWZQBjYv5cAw
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters
in_channel = 1
input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

def basic_test(model):
  model = model
  x = torch.randn(64,28,28).to(device)
  start = time.process_time()
  print(model(x).shape)
  end = time.process_time()
  print("time: ", end - start)

def chk_accuracy(loader, model):
    
  num_correct = 0
  num_samples = 0
  model.eval()
    
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device = device)
      y = y.to(device = device)
      scores = model(x)
      predictions = scores.argmax(1)
      num_correct += sum((predictions == y))
      num_samples += predictions.size(0)
            
    return float(num_correct)/float(num_samples)

def train(model):
  loss_fun = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate) 
  for epoch in range(num_epochs):
      model.train()
      if torch.cuda.is_available(): torch.cuda.empty_cache()
      model = model.to(device = device)

      loss_train = 0
      start = time.process_time()
      for batch, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device= device)
          
        #Forward Prop
        scores = model(data)
        loss = loss_fun(scores, targets)
          
        #Back prop
        optimizer.zero_grad()
        loss.backward()
        loss_train += loss.item()

        #Optimizer
        optimizer.step()

      train_acc = chk_accuracy(train_loader, model)
      val_acc = chk_accuracy(test_loader, model)
      avg_loss = loss_train/(len(train_loader))
      end = time.process_time()

      print('Epoch ({}/{}),Training loss : {:.4f}, Time: {:.2f}, train_accuracy:{:.4f}, val_accuracy:{:.4f}'.format(epoch+1, num_epochs, avg_loss, end - start, train_acc, val_acc))

  return model

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(RNN,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size*sequence_length, num_classes)


  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    out, _ = self.rnn(x, h0)
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    
    return out

model_RNN = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
basic_test(model_RNN)

class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(GRU,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size*sequence_length, num_classes)


  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    out, _ = self.gru(x, h0)
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    
    return out

model_GRU = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
basic_test(model_GRU)

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(LSTM,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size, num_classes)


  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0,c0))
    out = self.fc(out[:, -1, :])
    
    return out

model_LSTM = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
basic_test(model_LSTM)

class BLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(BLSTM,self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = True)
    self.fc = nn.Linear(hidden_size*2, num_classes)


  def forward(self, x):
    h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

    out, _ = self.lstm(x, (h0,c0))
    out = self.fc(out[:, -1, :])
    
    return out

model_BLSTM = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
basic_test(model_BLSTM)

#loading the data

X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = X.astype(np.float32)
y = np.int_(y)
X = X.reshape(X.shape[0], 28, 28)
print(X.shape, y.shape)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
y_tensor = y_tensor.type(torch.LongTensor)
X_train, X_test, y_train, y_test = train_test_split(X_tensor,y_tensor, test_size = (1/7), random_state = 42)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model_RNN = train(model_RNN)

model_GRU = train(model_GRU)

model_LSTM = train(model_LSTM)

model_BLSTM = train(model_BLSTM)