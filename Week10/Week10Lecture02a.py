from Week07Tuesdaya import *
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import confusion_matrix
from time import *


def drawConfusion(_myLabels, _myPredictions):
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

    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.train.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 784])
    yTensor = tf.placeholder(tf.float32, [None, 10])
    wTensor = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
    bTensor = tf.Variable(tf.zeros([1, 10], dtype=tf.float32))

    yHat = tf.matmul(xTensor, wTensor) + bTensor
    yHat = tf.nn.softmax(yHat)

    yHatLog = tf.log(yHat)
    yHatLogMultipliedY = yTensor * yHatLog

    yHatLogMultipliedY *= -1
    yHatLogMultipliedYAddColumn = tf.reduce_sum(yHatLogMultipliedY, axis=1)

    lastError = tf.reduce_mean(yHatLogMultipliedYAddColumn)

    optimizer = tf.train.GradientDescentOptimizer(.01).minimize(lastError)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for numOfIterations in range(250):
            sess.run(optimizer, feed_dict={xTensor: Xtrain,
                                           yTensor: Ytrain})

        allW = sess.run(wTensor)
        allB = sess.run(bTensor)

        firstTest = Xtest[0]
        result = np.matmul(firstTest, allW) + allB
        result = np.exp(result) / np.sum(np.exp(result))

        print(result)
        print(Ytest[0])


main()