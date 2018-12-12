from Week07Tuesdaya import *
import numpy as np
import math


def sigmoid(xList):
    result = []
    for el in xList:
        result.append(1/(1 + math.exp(-el)))
    return result

def derivativeSigmoid(xList):
    result = []
    for el in xList:
        result.append(el * (1 - el))
    return result

def main():

    x = np.arange(-10, 10, .2)
    print(x)
    y1 = sigmoid(x)
    print(y1)

    derivativeOfSigmoidX = derivativeSigmoid(y1)
    print("this is the derivative of sigmoid x", derivativeOfSigmoidX)

    newX = 10*x - 50
    print(newX)

    y2 = sigmoid(newX)

    f, axarr = plt.subplots(1,2, figsize=(10, 5))
    axarr[0].plot(x, y1)
    axarr[1].plot(x, y2)

    f.show()

main()