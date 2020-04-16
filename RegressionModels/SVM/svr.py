# SVR

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

# Feature Scaling
# As SVR model is not handling feature scaling, we need to manually handle this
# scenerio, we need to make train data and output data at same scale
from sklearn.preprocessing import StandardScaler
x_scaler= StandardScaler()
y_scaler= StandardScaler()
X= x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1,1))

# let us start with svm model ,which is very stright forward
from sklearn.svm import SVR
# we need to define kernal and default kernal is rbf only but we will define it for more clearity
svr= SVR(kernel='rbf')
# train model
svr.fit(X,y)
#let us predict value, but one thing is important that we need to transform input
inputValue=6.5
inputValueArr=np.array([[inputValue]])
predict=svr.predict(x_scaler.transform(inputValueArr))

predict = y_scaler.inverse_transform(predict)

plt.scatter(X,y,color='red')
plt.plot(X,svr.predict(X),color='green')
plt.title('Prediction Based on SVR')
plt.show()




