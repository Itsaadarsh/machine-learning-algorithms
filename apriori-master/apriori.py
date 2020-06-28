# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:10:01 2020

@author: @aadarshcodes
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
data = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(data.values[i,j]) for j in range(0,20)])
    
#Training the model
from apyori import apriori
rules = apriori(transactions,min_support = 0.003,min_confidence = 0.2,min_lift = 3,min_length = 2)

#Visualising the rules
result = []

for i in list(rules):

    result.append(["Base       -> " + str(i[2][0].items_base),

                    "add        -> " + str(i[2][0].items_add),

                    "Support    -> " + str(i[1]),

                    "Confidence -> " + str(i[2][0].confidence),

                    "Lift       -> " + str(i[2][0].lift)])
 