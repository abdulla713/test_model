#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# In[2]:


DEWA_solarPark = pd.read_csv('/home/mendel/test/test_model/code/file2+5inner.csv')
print('file2+5 shape = ', DEWA_solarPark.shape)


# In[3]:


DEWA_solarPark


# In[4]:



X = DEWA_solarPark[['Temperature','Global_Radiation','Humidity','Wind Direction','Wind Speed']]


# In[5]:


y = DEWA_solarPark['kWh']


# In[6]:



#train_test_split gives 4 parameters
#X_train and X_test to modify, y_train and y_test stay the same to compare with X later
#this will randomly split data to 80% training and 20% testing. can add parameter random_state=10 to have same samples
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[7]:


#Input contains NaN, infinity or a value too large for dtype('float64').
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

X_train_np[np.isnan(X_train_np)] = np.median(X_train_np[~np.isnan(X_train_np)])
y_train_np[np.isnan(y_train_np)] = np.median(y_train_np[~np.isnan(y_train_np)])
X_test_np[np.isnan(X_test_np)] = np.median(X_test_np[~np.isnan(X_test_np)])
y_test_np[np.isnan(y_test_np)] = np.median(y_test_np[~np.isnan(y_test_np)])

#classifier for plant 1 linear regression model
clf = RandomForestRegressor()
clf.fit(X_train_np, y_train_np)


# In[8]:


model_accuracy = clf.score(X_test_np, y_test_np)
print('model accuracy is {0:.2f}%'.format(model_accuracy*100))

