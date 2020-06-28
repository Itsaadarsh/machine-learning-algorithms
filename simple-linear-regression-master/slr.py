'''
DONE BY - AADARSH.S
IG - @aadarshcodes
'''

#Simple Regression Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Datasets
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values

#Spliting Test and Training Data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/4,random_state=0) 

'''
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
'''

#Fitting SLR to traing data
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,Y_train) 

#Predicting Test sets
Y_pred = regression.predict(X_test)

#Visualising the training data sets
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regression.predict(X_train),color = 'blue')
plt.title('Sal vs Exp (SLR)')
plt.xlabel("Exp")
plt.ylabel("Sal")
plt.show()

#Visualising the testing data sets
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regression.predict(X_train),color = 'blue')
plt.title('Sal vs Exp (SLR)')
plt.xlabel("Exp")
plt.ylabel("Sal")
plt.show()
