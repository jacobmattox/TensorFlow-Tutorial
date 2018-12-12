from Week07Tuesdaya import *
import numpy as np
import math
from random import *
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from time import *

def drawConfusion(_myLabels, _myPredictions):
    # old confusion_matrix
    # plt.imshow(returnedCM, cmap=plt.cm.Blues)
    # plt.xticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))
    # plt.yticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))
    # plt.xlabel("My Predicted")
    # plt.ylabel("Labels")
    # plt.savefig("handwritingConfusionMatrix")
    # new confusion_matrix with numbers
    returnedCM = confusion_matrix(_myLabels, _myPredictions)
    fig, ax = plt.subplots()
    ax.imshow(returnedCM)
    ax.set_xticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))
    ax.set_yticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))

    for i in range(len(returnedCM)):
        for j in range(len(returnedCM)):
            ax.text(j, i, returnedCM[i, j], ha="center", va="center", color="w")
    plt.savefig("handwritingConfustionMatrixUsingTensorFlowWith3HiddenLayers")
    plt.show()

def main():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(10000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 784])
    yTensor = tf.placeholder(tf.float32, [None, 10])


    wHiddenLayer1 = tf.Variable(tf.random_normal([784, 300]))
    bHiddenLayer1 = tf.Variable(tf.random_normal([1, 300]))

    outputOfFirst = tf.matmul(xTensor, wHiddenLayer1) + bHiddenLayer1
    outputOfFirst = tf.nn.sigmoid(outputOfFirst)

    wHiddenLayer2 = tf.Variable(tf.random_normal([300, 300]))
    bHiddenLayer2 = tf.Variable(tf.random_normal([1, 300]))

    outputOfSecond = tf.matmul(outputOfFirst, wHiddenLayer2) + bHiddenLayer2
    outputOfSecond = tf.nn.sigmoid(outputOfSecond)

    wHiddenLayer3 = tf.Variable(tf.random_normal([300, 300]))
    bHiddenLayer3 = tf.Variable(tf.random_normal([1, 300]))

    outputOfThird = tf.matmul(outputOfSecond, wHiddenLayer3) + bHiddenLayer3
    outputOfThird = tf.nn.sigmoid(outputOfThird)

    wOutputLayer = tf.Variable(tf.random_normal([300, 10]))
    bOutputLayer = tf.Variable(tf.random_normal([1, 10]))

    yHat = tf.matmul(outputOfThird, wOutputLayer) + bOutputLayer

    yHatSoftmax = tf.nn.softmax(yHat)

    lastError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yHat, labels=yTensor))

    #optimizer = tf.train.AdamOptimizer(1).minimize(lastError)
    optimizer = tf.train.GradientDescentOptimizer(2.5).minimize(lastError)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for numOfIterations in range(500):
            sess.run(optimizer, feed_dict={xTensor: Xtrain,
                                           yTensor: Ytrain})
            print(numOfIterations)

        resultTF = sess.run(yHatSoftmax, feed_dict={xTensor: Xtest})
        results = sess.run(yHat, feed_dict={xTensor: Xtest})
        print(results)

        indexOfHighest = np.argmax(resultTF, axis=1)
        indexOfActual = np.argmax(Ytest, axis=1)

        numOfCorrect = 0
        for idx, numOfIterations in enumerate(indexOfHighest):
            if numOfIterations == indexOfActual[idx]:
                numOfCorrect = numOfCorrect + 1
            print(numOfIterations, indexOfActual[idx])
        print(numOfCorrect)

        drawConfusion(indexOfActual, indexOfHighest)



main()
#gradientDescent 500 iterations at .00025 -- 435/500
#gradientDescent 500 iterations at .0003 -- 441/500
#gradientDescent 500 iterations at .00035 -- 433/500
#gradientDescent 500 iterations at .00031 -- 439/500
#gradientDescent 1000 iterations at .00001 -- 286/500
#gradientDescent 1000 iterations at .000025 -- 368/500
#gradientDescent 1000 iterations at .000035 -- 398/500
#gradientDescent 1000 iterations at .000030 -- 375/500
#gradientDescent 1000 iterations at .00005 -- 393/500
#gradientDescent 1000 iterations at .00004 -- 387/500
#gradientDescent 1000 iterations at .000006 -- 399/500
#gradientDescent 1000 iterations at .000009 -- 414/500
#gradientDescent 1000 iterations at .00000099 -- 415/500
#AdamDescent 1000 iterations at .18 -- 440/500
#AdamDescent 1000 iterations at .2 -- 444/500
#AdamDescent 1000 iterations at .25 -- 446/500
#AdamDescent 500 iterations at .30 -- 441/500
#AdamDescent 1000 iterations at .30 -- 443/500
#AdamDescent 1000 iterations at .275 -- 444/500