# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:05:01 2018

@author: Casey
This file just runs whichever features you specify. "Domain expertise"
"""

from Week07Lecture01ascaled import *
from sklearn import *
from sklearn.model_selection import cross_val_score

allX,allY,features = readFile([], [2])#enter indeces of features to run in first list

myModel = linear_model.LinearRegression()
myModel.fit(allX,allY)
scores = cross_val_score(myModel,allX,allY,scoring='neg_mean_squared_error', cv=10)#10 runs, each run has a different 90% of data for training, 10% for testing.
print(scores.mean())#take the mean score of all cross val runs