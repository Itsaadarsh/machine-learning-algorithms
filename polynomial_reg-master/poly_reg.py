'''
DONE BY - AADARSH.S
IG - @aadarshcodes
'''

#Polynomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Datasets
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

#Fitting Linear Regression into the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial regression into the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=8)   #Greater the degree more accurarte the predictions are!
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)


#Predicting
x_pred = np.reshape(float(input('Enter the level : ')) , newshape=(-1,1))
lin_reg.predict(x_pred)
lin_reg2.predict( poly_reg.fit_transform(x_pred))
print('The salary of level  {} is {}'.format(x_pred,lin_reg2.predict( poly_reg.fit_transform(x_pred))))

#Plotting 
x_grid = np.arange(min(X),max(X),0.1) #For HD plotting and smoother curves
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,Y,color = 'red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.xlabel('Level of exp')
plt.ylabel('salary')
plt.show()
