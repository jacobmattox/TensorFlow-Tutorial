
import random as r
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def readFile():
    myFile = open("kc_house_data.csv")
    allPrices=[]
    allSizes=[]
    allYearBuilt =[]

    allX=[]
    allY=[]

    for line in myFile.readlines()[1:]:
        try:
            currLine = line.split(',')

            allX.append([int(currLine[5]), int(currLine[14])])
            allY.append(int(currLine[2]))


            # allSizes.append(int(currLine[5]))
            # allYearBuilt.append(int(currLine[14]))
        except Exception as e:
            allX.pop()
            pass

    # myFigure = plt.figure()
    # ax = myFigure.add_subplot(111, projection='3d')
    # ax.scatter(allPrices, allSizes, allYearBuilt)
    # plt.show()

    return allX, allY

def getYHat(_XInstance, _W, _b):

    yHat = 0
    for indexInner, elInner in enumerate(_XInstance):

        yHat += elInner * _W[indexInner]
    yHat = float(yHat)
    yHat += _b
    return yHat

def calcError(_allX, _allY, _W, _b):
    totalError = 0

    for num, el in enumerate(_allX):
        totalError += (getYHat(el, _W, _b) - _allY[num]) ** 2

    return totalError/len(_allX)


def returnTwoPointsFromSlopeIntercept(_m, _b, _x1, _x2):
    y1 = _m * _x1 + _b
    y2 = _m * _x2 + _b
    return _x1, y1, _x2, y2

def main():

    allX, allY= readFile()

    # print(allX)
    # print(allY)
    allW = [10.0, 10.0]
    b = 10000.0

    #legacy from linear regression
    # minSize = min(allX)
    # maxSize = max(allX)
    #
    # minYearBuilt = min(allY)
    # maxYearBuilt = max(allY)
    #
    # fourPointsX = [minSize, minSize, maxSize, maxSize]
    # fourPointsY = [minYearBuilt, maxYearBuilt, minYearBuilt, maxYearBuilt]
    # fourPointsZ = []
    # print("fourpointsx: ", fourPointsX)
    #
    # for num in range(4):
    #     fourPointsZ.append(allW[0] * fourPointsX[num] + allW[1] * fourPointsY[num] + b)
    #
    # myFigure = plt.figure()
    # ax = myFigure.add_subplot(111, projection='3d')
    # ax.scatter(fourPointsX, fourPointsY, fourPointsZ)
    # plt.show()
    # return
    # return
    #
    # all_x = [r.randint(0,100) for i in range(100)]
    #
    # #print(all_x)
    #
    # all_y = []
    # for x in all_x:
    #     all_y.append(x * 5 * (r.random() + .5))
    #print(all_y)

    learningRate = .0000001


    #we want to calculate the derivative of error function with respect to M & B
    allXNumpy = np.array(allX)
    allWNumpy = np.array(allW)
    allWNumpy = allWNumpy.reshape(2, 1)
    allYNumpy = np.array(allY).reshape(len(allY),1)
    # print(allWNumpy)


    for el2 in range(1000):

        print(calcError(allX, allY, allWNumpy, b))
        der_respect_to_b = 0
        der_respect_to_w1 = 0
        der_respect_to_w2 = 0

        allYHat = np.matmul(allXNumpy, allWNumpy) + b
        allYHatMinusY = allYHat - allYNumpy

        X1numpy = allXNumpy[:, 0].reshape(20121, 1)
        X2numpy = allXNumpy[:, 1].reshape(20121, 1)

        afterMultX1 = allYHatMinusY * X1numpy
        afterMultAdditionX1 = np.sum(afterMultX1)

        afterMultX2 = allYHatMinusY * X2numpy
        afterMultAdditionX2 = np.sum(afterMultX2)

        der_respect_to_w1 = (afterMultAdditionX1 * 2) / len(allX)
        der_respect_to_w2 = (afterMultAdditionX2 * 2) / len(allX)
        der_respect_to_b = (np.sum(allYHatMinusY) * 2) / len(allX)
        allWNumpy[0] = allWNumpy[0] - learningRate * der_respect_to_w1
        allWNumpy[1] = allWNumpy[1] - learningRate * der_respect_to_w2
        b = b - learningRate * der_respect_to_b


        # for num, el in enumerate(allX):
        #     der_respect_to_w1 = der_respect_to_w1 + (el[0] * getYHat(el, allW, b) - allY[num])
        #     der_respect_to_w2 = der_respect_to_w2 + (el[1] * getYHat(el, allW, b) - allY[num])
        #     der_respect_to_b = der_respect_to_b + (getYHat(el, allW, b) - allY[num])
        #
        # der_respect_to_w1 = 2 * (der_respect_to_w1 / len(allX))
        # der_respect_to_w2 = 2 * (der_respect_to_w2 / len(allX))
        # der_respect_to_b = 2 * (der_respect_to_b / len(allX))
        #
        # allW[0] = allW[0] - learningRate * der_respect_to_w1
        # allW[1] = allW[1] - learningRate * der_respect_to_w2
        # b = b - learningRate * der_respect_to_b
        #
        #
        #
        #
        #     der_respect_to_m = der_respect_to_m + (all_y[num] - (initialM * all_x[num] + initialB)) * -1 * all_x[num]
        # der_respect_to_m = der_respect_to_m/50
        # der_respect_to_b = der_respect_to_b/50 * -1
        #
        # initialM = initialM - learningRate * der_respect_to_m
        # initialB = initialB - learningRate * der_respect_to_b

        # plt.scatter(all_x, all_y)
        #
        # x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)
        #
        # plt.plot([x1, x2], [y1, y2])
        # plt.savefig(str(el2) + '.png')

    # plt.scatter(all_x, all_y)
    #
    # x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)
    #
    # plt. plot([x1, x2], [y1, y2])

    # plt.show()

main()