# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:57:05 2020

@author: @aadarshcodes
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
data = pd.read_csv('Mall_Customers.csv')
x = data.iloc[:,[3,4]].values

#Dendrogram Graph (To find the optimal number of clusters)
import scipy.cluster.hierarchy as sch
dendrograms = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram Model')
plt.xlabel('Pts')
plt.ylabel('Euclidean Distance')
plt.show() 

#Applying hierarchical clustering 
from sklearn.cluster import AgglomerativeClustering
algo = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = algo.fit_predict(x)

#Visualising the HC
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 30, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 30, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 30, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 30, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 30, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-30)')
plt.legend()
plt.show()