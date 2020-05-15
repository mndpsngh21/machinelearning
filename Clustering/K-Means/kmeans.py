# K-Means Clustering
# This code is create cluster of observation based on their previous results

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


#Elow method to find optimal number of clusters
"""" 
There is below technique to find an optimal number of cluster which can be 
used to detect exact value of cluster which is requirement of KMeans cluster"""

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):   
 kmeans = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
 kmeans.fit(X)
 wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Clusters Count")
plt.ylabel(" WCSS")
plt.show() 
 
# so based on our observation for we have found 5 is optimal cluster
kmeans = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label='cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='green', label='cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='blue', label='cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='cyan', label='cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centers')
plt.title("K-Means Cluster")
plt.xlabel("Annual Income")
plt.ylabel("Spending")
plt.legend()
plt.show() 