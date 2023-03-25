#!/usr/bin/env python
# coding: utf-8

# Parkinson's disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages to it and affects more than 1 million individuals every year in India. This is chronic and has no ure yet. 

# XGBoost is a new ML algorithm designed with speed and perfomance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. 

# In this project I will import the XGBClassifier from XGBoost library: this is an implementation of scikit-learn API for XGBoost classification.
# 

# OBJECTIVE-----
# 
# To build a model to accurately detect the presence of Parkinson's disease in an individual.
# 
# 
# LIBRARIES
# 
# 1. scikit-learn
# 2. numpy
# 3. pandas
# 4. xgboost
# 
# DATASET
# 
# 1. UCI ML Parkisons dataset

# In[16]:


# install the required libraries

import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[19]:


# read the data into  a DataFrame

df = pd.read_csv("parkinsons.data")
df.head()


# In[20]:


# Get the features and labels

features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:,'status'].values


# In[21]:


# the status column has values 0 and 1 as labels:
# get the count of these labels for both 0 and 1
print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])


# In[22]:


# we have 147 ones and 48 zeros in the status column.


# In[23]:


# Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them.
# The MinMaxScaler transforms features by scaling them to a given range.
# The fit_transform() method fits to the data and them transfroms it.


scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels


# In[25]:


# Split the dataset into training and testing sets keeping 20% of the data for testing

x_train, x_test, y_train, y_test = train_test_split(
                                    x,y,test_size = 0.2,random_state = 7)


# In[26]:


# Initialize XGBClassifier and train the model.

model = XGBClassifier()
model.fit(x_train, y_train)


# In[27]:


# Generate y_pred(predicted values for x_test) and calculate the accuracy for the model

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred) * 100)


# 
# This gives an accuracy of 94.87%

# 
