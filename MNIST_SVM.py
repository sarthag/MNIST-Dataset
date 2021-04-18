#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


# In[2]:


X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = np.array(X)
y = np.int_(y)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = (1/7), random_state = 42)


# In[4]:


classifier = SVC(kernel='rbf',C=10)
classifier.fit(X_train, y_train) 
y_preds = classifier.predict(X_test)


# In[5]:


print(metrics.classification_report(y_test, y_preds))


# In[6]:


print(metrics.accuracy_score(y_test, y_preds))

