#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


# In[2]:


X,y = fetch_openml("mnist_784", version = 1, return_X_y = True)

X = np.array(X)
y = np.int_(y)


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = (1/7), random_state = 42)
print(len(X_train), len(X_test))

X_train1, X_train2, y_train1, y_train2 = X_train[:10000],X_train[10000:12000], y_train[:10000], y_train[10000:12000]
print(len(X_train1), len(X_train2))


# In[4]:


def euc_dist(a,b):
  dist = np.sqrt(np.sum((a-b)**2, axis = 1))
  return dist


# In[14]:


def KNN(X_known, X_test, y_known, k):
    predictions = []
    
    #distances = euc_dist_2d(X_known, X_test)


    for i in range(len(X_test)):
        distances = euc_dist(X_known, X_test[i])
        df = pd.DataFrame(np.transpose(distances), columns = ["Distances"])
        df["Values"] = np.transpose(y_known)

        df.sort_values(by = 'Distances', ascending = True, inplace = True)
        arr = np.array(df.head(k)["Values"].values)
        bin = np.zeros(10)
        for j in range(k):
            p = arr[j]
            bin[p] += 1
            
        if max(bin) == 1:
            predictions.append(arr[0])
            
        else:
            ans = bin.argmax() 
            predictions.append(ans)  


    return np.array(predictions)           


# In[15]:


def score(predictions, actual):
    ln = len(actual)
    scr = predictions - actual
    count = np.count_nonzero(scr)
  
    return 1 - count/ln


# In[16]:


for k in (3):
    preds = KNN(X_train1, X_train2, y_train1, k)
    scr = score(preds, y_train2)
    print("For k = ", k," score = ",scr)


# In[13]:


#k = 1 : score = 0.9585


# In[ ]:




