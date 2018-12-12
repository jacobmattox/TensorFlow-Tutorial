from Week07Tuesdaya import *
import numpy as np
import math
from random import *
import tensorflow as tf


def drawSigmoid(w, b):
    x = np.arange(100)
    y = x * w + b
    y = sigmoid(y)

    plt.plot(x, y, color='red')

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

    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]

    softmax = np.exp(z) / np.sum(np.exp(z))


    print(softmax)
main()