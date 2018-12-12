

from Week07Lecture01ascaled import *
from sklearn import *
from sklearn.model_selection import cross_val_score

allX, allY, features = readFile([3, 5, 13, 20, 12, 8], [2])

myModel = linear_model.LinearRegression()
myModel.fit(allX, allY)
scores = cross_val_score(myModel, allX, allY, scoring='neg_mean_squared_error', cv=10)
print(scores.mean())