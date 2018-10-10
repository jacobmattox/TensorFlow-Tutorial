

# prices = [349, 575, 295, 15990, 619.9, 468, 344.9, 350,
#           198.5, 6995, 450, 369.9, 399.9, 235, 230, 399,
#           869, 675, 350, 8875, 250, 1000, 450, 250, 7189]
# sqft = [1870, 1617, 1000, 17142, 2823, 2459, 2541, 2826,
#         1632, 7505, 2180, 1260, 1839, 1390, 1776, 1495, 3430,
#         3164, 2494, 15672, 1649, 3124, 2028, 1298, 8878]

#
# x = [8, 7, 10, 6]
# y = [16, 13, 15, 9]


import random as r
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def readFile():
    myFile = open("kc_house_data.csv")
    allPrices=[]
    allSizes=[]
    allYearBuilt =[]
    for line in myFile.readlines()[1:]:
        try:
            currLine = line.split(',')
            allPrices.append(int(currLine[2]))
            allSizes.append(int(currLine[5]))
            allYearBuilt.append(int(currLine[14]))
        except Exception as e:
            pass

    myFigure = plt.figure()
    ax = myFigure.add_subplot(111, projection='3d')
    ax.scatter(allPrices, allSizes, allYearBuilt)
    plt.savefig('3d')
    plt.show()

    return allSizes, allYearBuilt, allPrices

def calcError(_all_x1, _all_x2, _all_y, _w1, _w2, _b):
    my_sum = 0
    for num, el in enumerate(_all_x1):
        my_sum = my_sum + (_all_y[num] - (_w1 * _all_x1[num]) + _w2 * _all_x2[num] + _b) ** 2

    return my_sum/len(_all_x1)


def returnTwoPointsFromSlopeIntercept(_m, _b, _x1, _x2):
    y1 = _m * _x1 + _b
    y2 = _m * _x2 + _b
    return _x1, y1, _x2, y2

def main():

    all_x1, all_x2, all_y = readFile()
    w1 = 0
    w2 = 0
    b = 0

    minSize = min(all_x1)
    maxSize = max(all_x1)

    minYearBuilt = min(all_x2)
    maxYearBuilt = max(all_x2)

    fourPointsX = [minSize, minSize, maxSize, maxSize]
    fourPointsY = [minYearBuilt, maxYearBuilt, minYearBuilt, maxYearBuilt]
    fourPointsZ = []

    for num in range(4):
        fourPointsZ.append(w1 * fourPointsX[num] + w2 * fourPointsY[num] + b)


    print(calcError(all_x1, all_x2, all_y, w1, w2, b))


    return

    all_x = [r.randint(0,100) for i in range(100)]

    #print(all_x)

    all_y = []
    for x in all_x:
        all_y.append(x * 5 * (r.random() + .5))

    #print(all_y)

    learningRate = .00001
    initialM = 0
    initialB = 0

    #we want to calculate the derivative of error function with respect to M & B

    for el2 in range(100):
        der_respect_to_m = 0
        der_respect_to_b = 0
        for num, el in enumerate(all_x):
            der_respect_to_m = der_respect_to_m + (all_y[num] - (initialM * all_x[num] + initialB)) * -1 * all_x[num]
            der_respect_to_b = der_respect_to_b + (all_y[num] - (initialM * all_x[num] + initialB))
        der_respect_to_m = der_respect_to_m/50
        der_respect_to_b = der_respect_to_b/50 * -1

        initialM = initialM - learningRate * der_respect_to_m
        initialB = initialB - learningRate * der_respect_to_b

        plt.scatter(all_x, all_y)

        x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)

        plt.plot([x1, x2], [y1, y2])
        # plt.savefig(str(el2) + '.png')

    print(initialM, initialB)
    print("The final derivative with respect to m is ", der_respect_to_m)
    print("The final derivative with respect to b is ", der_respect_to_b)




    # plt.scatter(all_x, all_y)
    #
    # x1, y1, x2, y2 = returnTwoPointsFromSlopeIntercept(initialM, initialB, 0, 100)
    #
    # plt. plot([x1, x2], [y1, y2])

    print("The error of that line is", calcError(all_x, all_y, initialM, initialB))

    plt.savefig()
    plt.show()


main()