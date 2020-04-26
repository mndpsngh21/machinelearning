# K- Nearest Neighbor Regression
# This alogorithm works on nearest neignbor based approach, for a point any category 
# which have more neighbor will be considered as category of dependent variable
# most of part will remain same to Logistic Regression Algorithm only
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# let us create matrix for only age and salary those are present on index 2,3
X = dataset.iloc[:, [2, 3]].values
# let us load values those have actual dependent variable
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
# this will help to check performance of model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# As age and salary are not same on same scale so we need to do feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier
# We are using neighbors count =5 for categories data
# p=2 is for use euclidean distance
# metric is default metric used by classifier 
classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
"""
A confusion matrix is a table that is often used to describe the performance of
 a classification model (or “classifier”) on a set of test data
for which the true values are known
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# expected result from is [[64,4]
#                          [3 29]  ]
# which means only 4+3 are wrong predictions



# Visualising the Training set results
""""
Logic:
    1. create as mesh gride for all values those are present in feature matrix.
    2. predict the result for every pixel and set color of pixel based on result
       e.g. for N=red, Y= green
    3. define border value
    4. scatter all values based on result
""""
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()