from Week07Tuesdaya import *
from sklearn import *
#reads file
allX, allY = readFile([5], [2])

print(allX.shape)
print(allY.shape)
#does linear regression
myModel = linear_model.LinearRegression()
myModel.fit(allX, allY)
#these are w1 and b respectively
print(myModel.coef_)
print(myModel.intercept_)

x1 = 0
y1 = x1 * myModel.coef_[0][0] + myModel.intercept_[0]

x2 = 12000
y2 = x2 * myModel.coef_[0][0] + myModel.intercept_[0]

plt.scatter(allX, allY)
plt.plot([x1, x2], [y1, y2])
plt.show()