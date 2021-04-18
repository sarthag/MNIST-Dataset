#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Making all necessary imports
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


#Fetching the Dataset 
X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = np.array(X)
y = np.int_(y)


# In[3]:


#defining one hot encoding on y to simplify error checking 
def one_hot_encoding(y):
    one_hot_y = np.zeros((np.amax(y) +1, y.size))
    one_hot_y[y, np.arange(y.size), ] = 1
    return one_hot_y.T


# In[4]:


#performing one hot encoding
y = one_hot_encoding(y)


# In[5]:


#defining the activiation sigmoid function and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


# In[6]:


#defining the functions necessary fer training the network

#initializong weights and biases
def init_par():
    w1 = np.random.randn(25, 784)
    b1 = np.random.randn(25, 1)
    w2 = np.random.randn(10, 25)
    b2 = np.random.randn(10, 1)
    
    return w1, b1, w2, b2

#defining the forward prop a0 -> a1 -> a2
def fw_prop(w1, b1, w2, b2, X):
    a0 = X.T
    z1 = w1.dot(a0) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    
    return z1, a1, z2, a2


#defining backward prop to obtain parameters for updation
def bk_prop(z1, a1, z2, a2, w2, X, y):
    m = X.shape[0]
    p = 1/m
    
    da2 = (a2 - y.T)*deriv_sigmoid(z2)
    dw2 = p*da2.dot(a1.T)
    db2 = np.array([p*np.sum(da2)]).T
    
    da1 = w2.T.dot(da2)*deriv_sigmoid(z1) 
    dw1 = p*da1.dot(X)
    db1 = np.array([p*np.sum(da1)]).T
    
    return dw1, db1, dw2, db2


#function to update parameters
def upd_par(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    
    return w1, b1, w2, b2


# In[7]:


#Geadient descent function to train the neural net
def gradient_descent(x, Y, epochs, alpha):
    W1, b1 , W2, b2 = init_par()
    
    for i in range(epochs):
        Z1, A1, Z2, A2 = fw_prop(W1, b1 , W2, b2, x)
        dW1, db1, dW2, db2 = bk_prop(Z1, A1, Z2, A2, W2, x, Y)
        W1, b1 , W2, b2 = upd_par(W1, b1 , W2, b2, dW1, db1, dW2, db2, alpha)
        
    return W1, b1, W2, b2, A2


# In[8]:


#function to test the accuracy fot the NN for the test set
def testNN(X, W1, b1 , W2, b2):
    a0 = X
    a1 = sigmoid(np.dot(a0, W1) + b1)
    a2 = sigmoid(np.dot(a1 , W2) + b2)
    
    return a2    


# In[9]:


#function to determine the score of the neural net
def scoring(a2 ,y):
    print(y)
    print(a2)
    error = np.mean(np.abs(y - a2))
    score = 1 - error
    return score 


# In[10]:


#Dividing dataset into test and train
from sklearn.model_selection import train_test_split
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = (1/7), random_state = 42)
print(len(X_train), len(X_test))

alpha = 0.1
epochs = 1000


# In[11]:


#training the network to get necessary parameters
W1, b1, W2, b2, A2 = gradient_descent(X_train, y_train, epochs, alpha)


# In[12]:


#determining the train score
score = scoring(A2.T, y_train)
print('training score:', score)


# In[14]:


#determining the test score
a2_test = testNN(X_test, W1, W2, b1, b2)
score = scoring(a2_test.T, y_test)
print('test score:', score)

