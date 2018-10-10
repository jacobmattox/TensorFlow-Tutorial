
import matplotlib.pyplot as plt
import random as r
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import numpy as np


def readFile():
    myFile = open("kc_house_data.csv")
    allPrices = []
    allSizes = []
    allYearBuilt = []

    allX = []
    allY = []

    for line in myFile.readlines()[1:]:
        try:
            currLine = line.split(',')

            allX.append([int(currLine[5]), int(currLine[14])])
            allY.append(int(currLine[2]))

            #allSizes.append(int(currLine[5]))
            #allYearBuilt.append(int(currLine[14]))

        except Exception as e:
            #print(e)
            allX.pop()
            pass

    #print(allSizes)
    #print(allYearBuilt)
    #print(allPrices)

    return allX, allY

def getYHat(_XInstance, _W, _b):
    yHat = 0
    for indexInner, elInner in enumerate(_XInstance):
        yHat = yHat + elInner * _W[indexInner]
    yHat = yHat + _b

    return yHat

def calcError(_allX, _allY, _W, _b):

    totalError = 0

    for num, el in enumerate(_allX):
        totalError = totalError + (getYHat(el, _W, _b) - _allY[num])**2

        #my_sum = my_sum + (_all_y[num] - (_w1 * _all_x1[num] + _w2 * _all_x2[num] + _b)) ** 2

    return totalError / len(_allX)


def returnTwoPointsFromSlopeIntercept(_m, _b, _x1, _x2):

    y1 = _m * _x1 + _b
    y2 = _m * _x2 + _b

    return _x1, y1, _x2, y2






def main():

    allX, allY = readFile()

    print(len(allX))
    print(len(allY))

    b = 200000.0
    allW = [0.0, 0.0]


    myFigure = plt.figure()
    ax = myFigure.add_subplot(111, projection='3d')


    #ax.scatter(list(allX[:, 0]), list(allX[:, 1]), list(allY))


    return
    print('')

    #minSize = min(all_x1)
    #maxSize = max(all_x1)

    #minYearBuilt = min(all_x2)
    #maxYearBuilt = max(all_x2)

    #fourPointsX = [minSize, minSize, maxSize, maxSize]
    #fourPointsY = [minYearBuilt, maxYearBuilt, minYearBuilt, maxYearBuilt]
    #fourPointsZ = []

    #for num in range(4):
    #    fourPointsZ.append(w1 * fourPointsX[num] + w2 * fourPointsY[num] + b)

    #print(fourPointsY, fourPointsX, fourPointsZ)

    #ax.plot_trisurf(fourPointsX, fourPointsY, fourPointsZ, cmap='viridis', edgecolor='red')

    #plt.show()

    #print(minSize, maxSize, minYearBuilt, maxYearBuilt)



    #all_x = [r.randint(0, 100) for i in range(100)]

    #print(all_x)

    #print(r.random())

    #all_y = []
    #for x in all_x:
    #    all_y.append(x * 2 * (r.random() + .5))
    #print(all_y)

    learning_rate = .0000001

    allXnumpy = np.array(allX)
    allWnumpy = np.array(allW)
    allYnumpy = np.array(allY).reshape([len(allY), 1])



    print(allXnumpy)
    print(allWnumpy)
    allWnumpy = allWnumpy.reshape([2, 1])


    print(allXnumpy.shape)
    print(allWnumpy.shape)


    # We want to calculate the derivative of error function with respect to m
    #

    for el2 in range(10000):

        print(calcError(allX, allY, allWnumpy, b)[0])

        der_respect_to_w1 = 0
        der_respect_to_w2 = 0
        der_respect_to_b = 0

        allYHat = np.matmul(allXnumpy, allWnumpy) + b
        allYHatMinusY = allYHat - allYnumpy

        #print(allYHat.shape)
        #print(allYnumpy.shape)
        #print(allYHatMinusY.shape)

        #print(allX)

        X1numpy = allXnumpy[:, 0].reshape(20121, 1)
        X2numpy = allXnumpy[:, 1].reshape(20121, 1)

        #print(X1numpy.shape)

        afterMultX1 = allYHatMinusY * X1numpy
        afterMultX2 = allYHatMinusY * X2numpy

        afterMultAdditionX1 = np.sum(afterMultX1)
        afterMultAdditionX2 = np.sum(afterMultX2)

        der_respect_to_w1 = (afterMultAdditionX1 * 2) / len(allX)
        der_respect_to_w2 = (afterMultAdditionX2 * 2) / len(allX)
        der_respect_to_b = (np.average(allYHatMinusY) * 2) #/ len(allX)=

        allWnumpy[0] = allWnumpy[0] - learning_rate * der_respect_to_w1
        allWnumpy[1] = allWnumpy[1] - learning_rate * der_respect_to_w2
        b = b - (learning_rate * 10000) * der_respect_to_b

        print(allWnumpy)
        print(b)
        continue


        for num, el in enumerate(allX):



            der_respect_to_w1 = der_respect_to_w1 + (el[0] * (getYHat(el, allW, b) - allY[num]))
            der_respect_to_w2 = der_respect_to_w2 + (el[1] * (getYHat(el, allW, b) - allY[num]))
            der_respect_to_b = der_respect_to_b + (getYHat(el, allW, b) - allY[num])

            #der_respect_to_m = der_respect_to_m + (all_y[num] - (initialM * all_x[num] + initialB)) * -1 * all_x[num]

        der_respect_to_w1 = 2* (der_respect_to_w1 / len(allX))
        der_respect_to_w2 = 2 * (der_respect_to_w2 / len(allX))
        der_respect_to_b = 2 * (der_respect_to_b / len(allX))

        allW[0] = allW[0] - learning_rate * der_respect_to_w1
        allW[1] = allW[1] - learning_rate * der_respect_to_w2
        b = b - learning_rate * der_respect_to_b



        #initialM = initialM - learning_rate * der_respect_to_m
        #print(initialM)
        #print(der_respect_to_m)
        #x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)
        #plt.plot([x1, x2], [y1, y2])
        #plt.scatter(all_x, all_y)
        #plt.savefig(str(el2))
        #plt.clf()
        #plt.close()

    #print("The final der with respect to m is", der_respect_to_m)

    #plt.scatter(all_x, all_y)

    #x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)

    #plt.plot([x1, x2], [y1, y2])

    #print("The error of that line is", calcError(all_x, all_y, initialM, initialB))

    #plt.show()


main()