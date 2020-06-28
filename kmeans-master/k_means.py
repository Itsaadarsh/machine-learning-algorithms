# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:58:20 2020

@author: @aadarshcodes
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
data = pd.read_csv('total_data_na.csv')
x = data.iloc[100:,[15,17]].values

#Elbow Method for clustering
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=500,random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.show()

#Applying Kmeans
kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=42)
y_means=kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_means==0,0],x[y_means==0,1],s=30,c='blue',label='2')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=30,c='yellow',label='3')
plt.scatter(x[y_means==2,0],x[y_means==2,1],s=30,c='red',label='1')
#plt.scatter(x[y_means==3,0],x[y_means==3,1],s=30,c='green',label='Target 2')
#plt.scatter(x[y_means==4,0],x[y_means==4,1],s=30,c='black',    ='Target 1')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='orange')
plt.title('IPL 2018 ')
plt.xlabel('Overs bowled')
plt.ylabel('Wickets taken')
plt.legend()
plt.show()
