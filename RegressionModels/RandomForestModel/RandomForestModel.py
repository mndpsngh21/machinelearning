# Practicins code for Random Forest Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# no feature scaling is require for Random Forest Model
"""
# Feature Scaling
# As SVR model is not handling feature scaling, we need to manually handle this
# scenerio, we need to make train data and output data at same scale
from sklearn.preprocessing import StandardScaler
x_scaler= StandardScaler()
y_scaler= StandardScaler()
X= x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))"""

# let us start with Random Forest model ,which is very stright forward
from sklearn.ensemble import RandomForestRegressor
# we are defining random state as we need same value everytime which is similar
# to Decision Tree but one thing is different from decision tree is that we need 
# to define n_estimators which is size of forest, we can fine tune this value for better result
regressor= RandomForestRegressor(n_estimators=300,random_state=0)
# train model
regressor.fit(X,y)
#let us predict value, but one thing is important that we need to transform input
predict=regressor.predict(np.array([[6.5]]))

# Let us view this is higher resolution this will be similar to decision tree only more steps can be observed
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# so, we are done with decision tree



