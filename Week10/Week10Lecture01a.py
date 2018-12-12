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

    seed(0)
    allX = np.arange(0, 100)
    allY = np.zeros((100, ))

    for idx, el in enumerate(allX):
        myRandom = randint(0, 100)
        if myRandom < idx:
            allY[idx] = 1

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w = 0.0
    b = 0.0

    wTensor = tf.Variable(w, name="weight")
    bTensor = tf.Variable(b, name="intercept")

    Y_hat = X * wTensor + bTensor

    Y_hat = tf.nn.sigmoid(Y_hat)

    error = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,
                                                    logits=Y_hat)

    error = tf.reduce_mean(error)

    #optimizer = tf.train.AdamOptimizer(.01). minimize(error)
    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(error)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for el in range(20000):
            sess.run(optimizer, feed_dict={X: allX, Y: allY})

            print("The value of w", sess.run(wTensor))
            print("The value of b", sess.run(bTensor))

        plt.scatter(allX, allY)
        drawSigmoid(sess.run(wTensor), sess.run(bTensor))
        plt.show()

main()