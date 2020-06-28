# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 01:34:40 2020

@author: @aadarshcodes
"""

import numpy as np
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
label_1 = LabelEncoder()
x[:,1] = label_1.fit_transform(x[:,1])
label_2 = LabelEncoder()
x[:,2] = label_2.fit_transform(x[:,2])
onehotencoder = ColumnTransformer(transformers=[('Geography',OneHotEncoder(),[1])],remainder='passthrough')
x = onehotencoder.fit_transform(x)
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 69)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(output_dim = 12,activation='relu',init='uniform',input_dim=11))
model.add(Dense(output_dim = 12,activation='relu',init='uniform'))
model.add(Dense(output_dim = 12,activation='relu',init='uniform'))
#model.add(Dense(output_dim = 6,activation='relu',init='uniform'))
model.add(Dense(output_dim = 1,activation='sigmoid',init='uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, nb_epoch = 400)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
