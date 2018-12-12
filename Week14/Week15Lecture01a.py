import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def main():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 28, 28, 1])

    myFilter = [[[[.5, .5]], [[.5, -.5]]],
                [[[-.5, .5]], [[-.5, -.5]]]]

    myFilter = np.array(myFilter)

    filterTensor = tf.placeholder(tf.float32, [2, 2, 1, 2])

    layer = tf.nn.conv2d(input=xTensor,
                         filter=filterTensor,
                         strides=[1, 1, 1, 1],
                         padding="SAME")

    layerAfterAfterPooling = tf.nn.max_pool(layer,
                                            [1, 2, 2, 1],
                                            [1, 2, 2, 1],
                                            "SAME")

    layerAfterAfterPoolingRelu = tf.nn.relu(layerAfterAfterPooling)

    newSize = layerAfterAfterPoolingRelu.shape[1] * layerAfterAfterPoolingRelu.shape[2] * layerAfterAfterPoolingRelu.shape[3]
    print(newSize)

    afterBeingFlattened = tf.reshape(layerAfterAfterPoolingRelu, [-1, newSize])
    print(afterBeingFlattened.shape)

    return

    #print(layerAfterAfterPooling.shape)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        result = sess.run(layer, feed_dict={xTensor: Xtrain[0:5].reshape(5, 28, 28, 1),
                                            filterTensor: myFilter})

        resultOfPooling = sess.run(layerAfterAfterPooling, feed_dict={xTensor: Xtrain[0:5].reshape(5, 28, 28, 1),
                                                                      filterTensor: myFilter})

        resultOfPoolingRelu = sess.run(layerAfterAfterPoolingRelu, feed_dict={xTensor: Xtrain[0:5].reshape(5, 28, 28, 1),
                                                                              filterTensor: myFilter})

        im1before = Xtrain[3].reshape(28, 28)

        im1after1 = result[3, :, :, 0]
        im1after2 = np.abs(result[3, :, :, 1])

        f, axarr = plt.subplots(3, 3)

        axarr[0, 0].imshow(im1before, cmap="gray")
        axarr[0, 1].imshow(im1after1, cmap="gray")
        axarr[0, 2].imshow(im1after2, cmap="gray")

        axarr[1, 0].imshow(resultOfPooling[3, :, :, 0], cmap="gray")
        axarr[1, 1].imshow(np.abs(resultOfPooling[3, :, :, 0]), cmap="gray")
        axarr[1, 2].imshow(resultOfPoolingRelu[3, :, :, 0], cmap="gray")

        r = np.abs(resultOfPooling[3, :, :, 0]) - resultOfPoolingRelu[3, :, :, 0]
        print(r)

        axarr[2, 0].imshow(r, cmap="gray")


        r = np.abs(r)
        r = np.sum(r, axis=0)
        r = np.sum(r, axis=0)
        print(r)

        print()

        f.show()

main()