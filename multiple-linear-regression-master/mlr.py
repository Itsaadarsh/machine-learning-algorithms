'''
DONE BY - AADARSH.S
IG - @aadarshcodes
'''

#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Categorical data encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Dummy variable removal
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

#Fitting MLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the model with test set
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.api as sm
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog = X_opt).fit()
regressor_OLS.summary()


