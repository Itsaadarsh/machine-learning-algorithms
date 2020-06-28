# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 22:33:25 2020

@author: @aadarshcodes
"""

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#UBC Algorithm
import math
N = 10000
d = 10
ads_sel = []
total_reward = 0
no_of_sel = [0] * d
sum_of_reward = [0] * d
for n in range(N):
    ad = 0
    max_up = 0
    for i in range(d):
        if(no_of_sel[i] > 0):
            avg_reward = sum_of_reward[i] / no_of_sel[i]
            del_i = math.sqrt(3/2 * math.log(n + 1) / no_of_sel[i])
            upper_bound = avg_reward + del_i
        else: 
            upper_bound = 1e400
        if(upper_bound > max_up):
            max_up = upper_bound
            ad = i
    ads_sel.append(ad)
    no_of_sel[ad] = no_of_sel[ad]+ 1
    reward = data.values[n,ad]
    sum_of_reward[ad] += reward
    total_reward += reward
    
#Visualisation 
plt.hist(ads_sel)
