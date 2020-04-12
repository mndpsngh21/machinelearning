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

# now we have predicted all values but there is one problem we are considering 
# all independent variables which effects our model performance, so we will use 
# backward elimination technique to find variables those are effecting our 
# prediction.

#To build optimal model we will use one library, which has this feature inbuilt
import statsmodels.formula.api as sm
""" we need to understand one thing that we do not have any constant variable
    which is required for expression as below:
        y= b0 + b1.x1 + b2.x2 +......+bn.xn
        y= profit
        x1,x2,x3...xn= independent varabiables
        b0,b1,b2...bn= coefficient of variables
        b0= constant ---> which is unknown for us
    So what is solution? we will fill b0 with value=1 to handle this problem
, we are doing this for this library's requirement """
# we are going to use np library to fill this values
x_data = np.append(arr=np.ones((50,1)).astype(int),values = x_data, axis=1)

# Now we have data but for elamination we need to find variables those are effecting 
# our model

# For model optimization we will use another library and make our model optimal
""" Backword elimination techinque:
    1. set significance level S i.e. 0.05 for our case 
    2. fit full model with all possible predictors.
    3. find predictor with highest p value and if p>S next step,otherwise 
       finish. 
    4. remove predictor
    5. go to 2 """"
    
# we will use sm for fitting
# we are making matric for optimization and will perform operation on that    
X_opt =x_data[:,[0,1,2,3,4]]
# let us fit this model for checking performance
import statsmodels.regression.linear_model as lm
regressor_OLS = lm.OLS(endog = y_data, exog = X_opt).fit()
# execute below command and look for P>|t| value which has highest values remove that 
regressor_OLS.summary()     
# we found x2 has 0.991 which is more than significance level S i.e. 0.05
# let us remove it
X_opt =x_data[:,[0,1,3,4,5]]
# let us fit this model for checking performance
regressor_OLS = lm.OLS(endog = y_data, exog = X_opt).fit()
# execute below command and look for P>|t| value which has highest values remove that 
regressor_OLS.summary()     
# we found x1 has 0.703 which is more than significance level S i.e. 0.05
# let us remove it
X_opt =x_data[:,[0,3,4,5]]
# let us fit this model for checking performance
regressor_OLS = lm.OLS(endog = y_data, exog = X_opt).fit()
# execute below command and look for P>|t| value which has highest values remove that 
regressor_OLS.summary()     
# we found x1 has 0.602 which is more than significance level S i.e. 0.05
X_opt =x_data[:,[0,3,5]]
# let us fit this model for checking performance
regressor_OLS = lm.OLS(endog = y_data, exog = X_opt).fit()
# execute below command and look for P>|t| value which has highest values remove that 
regressor_OLS.summary() 
# we found x1 has 0.060 which is more than significance level S i.e. 0.05
# let us remove it    
X_opt =x_data[:,[0,3]]
# let us fit this model for checking performance
regressor_OLS = lm.OLS(endog = y_data, exog = X_opt).fit()
# execute below command and look for P>|t| value which has highest values remove that 
regressor_OLS.summary()     
# so now we dont have anything to remove so we will exit


### automatic elimination technique

import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as lm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = lm.OLS(y_data, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
    
 
SL = 0.05
X_opt = x_data[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)    
       

