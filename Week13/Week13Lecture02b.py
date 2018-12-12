import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def main():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 28, 28, 1])

    myFilter = [[[[.5]], [[-.5]]],
                [[[.5]], [[-.5]]]]

    myFilter = np.array(myFilter)

    filterTensor = tf.placeholder(tf.float32, [2, 2, 1, 1])

    layer = tf.nn.conv2d(input=xTensor,
                         filter=filterTensor,
                         strides=[1, 1, 1, 1],
                         padding="VALID")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        result = sess.run(layer, feed_dict={xTensor: Xtrain[0:5].reshape(5, 28, 28, 1),
                                            filterTensor: myFilter})

        print(result.shape)

        im1before = Xtrain[3].reshape(28, 28)
        im1after = np.abs(result[3].reshape(27, 27))

        f, axarr = plt.subplots(1, 2)

        axarr[0].imshow(im1before, cmap="gray")
        axarr[1].imshow(im1after, cmap="gray")

        f.show()
main()