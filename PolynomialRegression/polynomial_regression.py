# Polynomial Regression
"""
Polynomial Regression is for use cases when things change exponetionally, It has
polynomial curve for code
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset from files
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Let us process simple regression model for processing
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,y)


#Let us start with Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
#we have used degree=4 which is best fit for our model but you can try different
#values for degree to fing proper value
poly_feat =PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X,y)
lr_2= LinearRegression()
lr_2.fit(X_poly,y)


# Visualising information in graph
plt.scatter(X,y, color='red')
# Greeen color line is showing simple Linear Regression
plt.plot(X,lin_reg.predict(X),color='green')
# blue color is showing our model in polynomial regression form
plt.plot(X,lr_2.predict(poly_feat.fit_transform(X,y)),color='blue')
plt.title('Comparison of Linear(green) and Polynomial(blue) Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

