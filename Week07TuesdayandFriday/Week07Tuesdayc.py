from Week07Tuesdaya import *
from sklearn import *
#reads file
allX, allY = readFile([5, 14], [2])

print(allX.shape)
print(allY.shape)
#does linear regression
myModel = linear_model.LinearRegression()
myModel.fit(allX, allY)
#these are w1 and b respectively
print(myModel.coef_)
print(myModel.intercept_)
#for plotting in 3d
myFigure = plt.figure()
ax = myFigure.add_subplot(111, projection='3d')
ax.scatter(allX[:, 0], allX[:, 1], allY)


minSize = min(allX[:, 0])
maxSize = max(allX[:, 0])

minYear = min(allX[:, 1])
maxYear = max(allX[:, 1])

point1X = np.array([minSize, minYear]).reshape(2, 1)
point1Y = np.sum(point1X * myModel.coef_.transpose()) + np.asscalar(myModel.intercept_)

point2X = np.array([minSize, maxYear]).reshape(2, 1)
point2Y = np.sum(point2X * myModel.coef_.transpose()) + myModel.intercept_[0]#np.asscalar(myModel.intercept_))

point3X = np.array([maxSize, minYear]).reshape(2, 1)
point3Y = np.sum(point3X * myModel.coef_.transpose()) + myModel.intercept_[0]#np.asscalar(myModel.intercept_))

point4X = np.array([maxSize, maxYear]).reshape(2, 1)
point4Y = np.sum(point4X * myModel.coef_.transpose()) + myModel.intercept_[0] #np.asscalar(myModel.intercept_))

allFourX1 = [point1X[0][0], point2X[0][0], point3X[0][0], point4X[0][0]]
allFourX2 = [point1X[1][0], point2X[1][0], point3X[1][0], point4X[1][0]]
allFourY = [point1Y, point2Y, point3Y, point4Y]

ax.plot_trisurf(allFourX1,
                allFourX2,
                allFourY,
                cmap='viridis')


# for el in range(0, 361, 10):
#     #print(el)
#     ax.view_init(10, el)

myFigure.show()


