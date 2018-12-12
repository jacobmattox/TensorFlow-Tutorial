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
    plt.savefig("handwritingConfustionMatrixUsingSoftMax")
    plt.show()

def main():
    timeStart = time()
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(60000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 784])
    yTensor = tf.placeholder(tf.float32, [None, 10])

    wTensor = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
    bTensor = tf.Variable(tf.zeros([1, 10], dtype=tf.float32))

    yHat = tf.matmul(xTensor, wTensor) + bTensor
    yHat = tf.nn.softmax(yHat)

    yHatLog = tf.log(yHat)
    yHatLogMultipliedY = yTensor * yHatLog
    yHatLogMultipliedY = -1 * yHatLogMultipliedY

    yHatLogMultipliedYAddColumn = tf.reduce_sum(yHatLogMultipliedY, axis=1)

    lastError = tf.reduce_mean(yHatLogMultipliedYAddColumn)

    #calculates all of last error without needint anything below the first yHat in this code
    #lastError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yHat, labels=yTensor))

    optimizer = tf.train.GradientDescentOptimizer(.01).minimize(lastError)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for numOfIterations in range(30000):
            sess.run(optimizer, feed_dict={xTensor: Xtrain,
                                           yTensor: Ytrain})
            print(numOfIterations)
        allW = sess.run(wTensor)
        allB = sess.run(bTensor)
        resultTF = sess.run(yHat, feed_dict={xTensor: Xtest})

        firstTest = Xtest[0]
        result = np.matmul(firstTest, allW) + allB
        result = np.exp(result) / np.sum(np.exp(result))

        indexOfHighest = np.argmax(resultTF, axis=1)
        indexOfActual = np.argmax(Ytest, axis=1)
        numOfCorrect = 0
        for idx, el in enumerate(indexOfHighest):
            if el == indexOfActual[idx]:
                numOfCorrect = numOfCorrect + 1

        timeStop = time()
        print(timeStop - timeStart)
        print(numOfCorrect)
        #drawConfusion(indexOfHighest, indexOfActual)

        #30k iterations 60k training
        #2853.9427468776703
        #465/500
        # 30k iterations 10k training
        # 456.3257968425751
        # 465/500
        #20k iterations 10k training
        # 248.36328983306885
        # 462/500
        #10k iterations 60k training
        # 1134.3759
        # 457/500
main()