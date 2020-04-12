# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:27:33 2020
Difference between simple linear model and this model is that simple Linear 
Regression model and only one independent variable but for this model we have
multiple independent variable, we will use a sample data of startup companies 
and try predict profit for a company
@author: Mandeep
"""

# Sample project for MultiLinear Regression

# Import required libraries 
import numpy as np
# library for draw graphs
import matplotlib.pyplot as plt
# panda is library for data read from csv
import pandas as pd

# Importing data from csv file
dataset = pd.read_csv('50_Startups.csv')
# dataSet for read all values those are indpendent variables 
x_data = dataset.iloc[:, :-1].values
# dataSet for read for all dependent variable i.e which we want to predict the data 
# for this case we are going to predict profit and 
y_data = dataset.iloc[:, 4].values

# As in data we have categorial information we will need to encode that data to
# some meaningfulo information. The reason behind this is that we can only use
# mathematical values but we have data which has text values    


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# we need to use index 3 which is basically column of city
x_data[:, 3] = labelencoder_X.fit_transform(x_data[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x_data = onehotencoder.fit_transform(x_data).toarray()


# We need to avoid dummy variable trap, For this Read about Dummy Variable Trap
#By default library handle this, it is for our learning purpose 
# we are removing first column
x_data =x_data[:,1:]

# For testing purpose we will split data into two part one for training purpose and another for testing purpose
# We will use sklearn model_selection for split data into test and training part
# X_train - trainning data set
# X_test - testing data set
# y_train - training dependent variables (we are resuts for data)
# y_test - testing data actual results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 0)


# Now we will fit model and we will use linear regression model for it
# we will use same model and simple linear regression model for this functionality
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

# So finally let us predict profit for our test_data
test_predict=regressor.predict(X_test)

