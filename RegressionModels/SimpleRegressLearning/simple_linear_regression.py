# Sample project for Linear Regression

# Import required libraries 
import numpy as np
# library for draw graphs
import matplotlib.pyplot as plt
# panda is library for data read from csv
import pandas as pd

# Importing data from csv file
dataset = pd.read_csv('Salary_Data.csv')
# dataSet for read all values those are indpendent variables 
x_data = dataset.iloc[:, :-1].values
# dataSet for read for all dependent variable i.e which we want to predict the data
y_data = dataset.iloc[:, 1].values

# For testing purpose we will split data into two part one for training purpose and another for testing purpose
# We will use sklearn model_selection for split data into test and training part
# X_train - trainning data set
# X_test - testing data set
# y_train - training dependent variables (we are resuts for data)
# y_test - testing data actual results
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 1/3, random_state = 0)

# This is starting of our machine we will use linear model
# we will use LinearRegression class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# by calling we will train our machine
regressor.fit(X_train, y_train)

# Predicting the Test set results,machine will return data as array
y_pred = regressor.predict(X_test)

# For understanding values and view data in contrast of linear regression values 
plt.scatter(X_train, y_train, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary/Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results to view on what basis machine is predicting value
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()