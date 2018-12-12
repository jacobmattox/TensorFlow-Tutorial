import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import *

def main():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    yTensor = tf.placeholder(tf.float32, shape=[None, 10])

    firstLayerShape = [5, 5, 1, 16]
    firstLayerWeights = tf.Variable(initial_value=tf.random_normal(firstLayerShape), dtype=tf.float32)

    firstLayer = tf.nn.conv2d(input=xTensor,
                              filter=firstLayerWeights,
                              strides=[1, 1, 1, 1],
                              padding="SAME")



    firstLayerAfterPooling = tf.nn.max_pool(value=firstLayer,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding="SAME")

    firstLayerAfterPoolingRelu = tf.nn.relu(firstLayerAfterPooling)

    secondLayerShape = [5, 5, 16, 36]
    secondLayerWeights = tf.Variable(initial_value=tf.random_normal(secondLayerShape), dtype=tf.float32)

    secondLayer = tf.nn.conv2d(input=firstLayerAfterPoolingRelu,
                              filter=secondLayerWeights,
                              strides=[1, 1, 1, 1],
                              padding="SAME")


    secondLayerAfterPooling = tf.nn.max_pool(value=secondLayer,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding="SAME")

    secondLayerAfterPoolingRelu = tf.nn.relu(secondLayerAfterPooling)


    numOfFeatures = secondLayerAfterPoolingRelu.shape[1] * secondLayerAfterPoolingRelu.shape[2] * secondLayerAfterPoolingRelu.shape[3]
    print(numOfFeatures)
    layer_flat = tf.reshape(secondLayerAfterPoolingRelu, [-1, numOfFeatures])

    firstFlatLayerWeights = tf.Variable(initial_value=tf.random_normal([int(numOfFeatures), 128]), dtype=tf.float32)
    firstFlatLayerBiases = tf.Variable(initial_value=tf.random_normal([128]), dtype=tf.float32)
    firstFlatLayerOut = tf.matmul(layer_flat, firstFlatLayerWeights) + firstFlatLayerBiases

    secondFlatLayerWeights = tf.Variable(initial_value=tf.random_normal([128, 10]), dtype=tf.float32)
    secondFlatLayerBiases = tf.Variable(initial_value=tf.random_normal([10]), dtype=tf.float32)
    secondFlatLayerOut = tf.matmul(firstFlatLayerOut, secondFlatLayerWeights) + secondFlatLayerBiases

    yPredicted = tf.nn.softmax(secondFlatLayerOut)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=secondFlatLayerOut,
                                                            labels=yTensor)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(.001).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)


        for el in range(300):

            allNums = list(range(50000))
            shuffle(allNums)
            allNums = allNums[:100]

            sess.run(optimizer, feed_dict={xTensor: Xtrain[allNums].reshape(100, 28, 28, 1),
                                           yTensor: Ytrain[allNums]})

            print(el)
            if el == 0:
                weightsResult = sess.run(firstLayer, feed_dict={xTensor: Xtrain[:1].reshape(1, 28, 28, 1)})
                weightsResult1 = weightsResult[0, :, :, 0]
                weightsResult2 = weightsResult[0, :, :, 1]
                weightsResult3 = weightsResult[0, :, :, 2]
                weightsResult4 = weightsResult[0, :, :, 3]

                print(np.array(weightsResult1))

                f, axarr = plt.subplots(2, 2)
                axarr[0, 0].imshow(weightsResult1, cmap="gray")
                axarr[0, 1].imshow(weightsResult2, cmap="gray")
                axarr[1, 0].imshow(weightsResult3, cmap="gray")
                axarr[1, 1].imshow(weightsResult4, cmap="gray")
                f.show()

        weightsResult = sess.run(firstLayer, feed_dict={xTensor: Xtrain[:1].reshape(1, 28, 28, 1)})
        weightsResult1 = weightsResult[0, :, :, 0]
        weightsResult2 = weightsResult[0, :, :, 1]
        weightsResult3 = weightsResult[0, :, :, 2]
        weightsResult4 = weightsResult[0, :, :, 3]

        print(np.array(weightsResult1))

        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(weightsResult1, cmap="gray")
        axarr[0, 1].imshow(weightsResult2, cmap="gray")
        axarr[1, 0].imshow(weightsResult3, cmap="gray")
        axarr[1, 1].imshow(weightsResult4, cmap="gray")
        f.show()


        result = sess.run(yPredicted, feed_dict={xTensor: Xtest[:200].reshape(200, 28, 28, 1)})
        print(np.argmax(result, axis=1))

        print("Actual labels:")
        print(np.argmax(Ytest[:200], axis=1))

        numOfCorrect = 0
        for idx, el in enumerate(Ytest[:200]):
            if np.argmax(el) == np.argmax(result, axis=1)[idx]:
                numOfCorrect = numOfCorrect + 1

        print(numOfCorrect)


main()