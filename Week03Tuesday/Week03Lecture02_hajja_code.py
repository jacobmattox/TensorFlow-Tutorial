
import matplotlib.pyplot as plt
import random as r
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

def readFile():
    myFile = open("kc_house_data.csv")
    allPrices = []
    allSizes = []
    allYearBuilt = []
    for line in myFile.readlines()[1:]:
        try:
            currLine = line.split(',')
            allPrices.append(int(currLine[2]))
            allSizes.append(int(currLine[5]))
            allYearBuilt.append(int(currLine[14]))
        except Exception as e:
            print(e)
            pass

    print(allSizes)
    print(allYearBuilt)
    print(allPrices)

    return allSizes, allYearBuilt, allPrices

def calcError(_all_x1, _all_x2, _all_y, _w1, _w2, _b):
    my_sum = 0

    for num, el in enumerate(_all_x1):
        my_sum = my_sum + (_all_y[num] - (_w1 * _all_x1[num] + _w2 * _all_x2[num] + _b)) ** 2

    return my_sum / len(_all_x1)

def returnTwoPointsFromSlopeIntercept(_m, _b, _x1, _x2):

    y1 = _m * _x1 + _b
    y2 = _m * _x2 + _b

    return _x1, y1, _x2, y2






def main():
    all_x1, all_x2, all_y = readFile()

    myFigure = plt.figure()
    ax = myFigure.add_subplot(111, projection='3d')
    ax.scatter(all_x1, all_x2, all_y)



    w1 = 10
    w2 = 10
    b = 10

    minSize = min(all_x1)
    maxSize = max(all_x1)

    minYearBuilt = min(all_x2)
    maxYearBuilt = max(all_x2)

    fourPointsX = [minSize, minSize, maxSize, maxSize]
    fourPointsY = [minYearBuilt, maxYearBuilt, minYearBuilt, maxYearBuilt]
    fourPointsZ = []

    for num in range(4):
        fourPointsZ.append(w1 * fourPointsX[num] + w2 * fourPointsY[num] + b)

    print(fourPointsY, fourPointsX, fourPointsZ)

    ax.plot_trisurf(fourPointsX, fourPointsY, fourPointsZ, cmap='viridis', edgecolor='red')
    plt.savefig('3d_with_plane')
    plt.show()

    print(minSize, maxSize, minYearBuilt, maxYearBuilt)

    print(calcError(all_x1, all_x2, all_y, w1, w2, b))

    return


    return


    all_x = [r.randint(0, 100) for i in range(100)]

    print(all_x)

    print(r.random())

    all_y = []
    for x in all_x:
        all_y.append(x * 2 * (r.random() + .5))

    print(all_y)

    learning_rate = .00001
    initialM = 0
    initialB = 0

    # We want to calculate the derivative of error function with respect to m
    #

    for el2 in range(100):

        der_respect_to_m = 0
        for num, el in enumerate(all_x):
            der_respect_to_m = der_respect_to_m + (all_y[num] - (initialM * all_x[num] + initialB)) * -1 * all_x[num]

        der_respect_to_m = der_respect_to_m / 50

        initialM = initialM - learning_rate * der_respect_to_m
        print(initialM)
        print(der_respect_to_m)
        x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)
        plt.plot([x1, x2], [y1, y2])
        plt.scatter(all_x, all_y)
        plt.savefig(str(el2))
        plt.clf()
        plt.close()

    print("The final der with respect to m is", der_respect_to_m)

    plt.scatter(all_x, all_y)

    x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)

    plt.plot([x1, x2], [y1, y2])

    print("The error of that line is", calcError(all_x, all_y, initialM, initialB))

    plt.show()


main()