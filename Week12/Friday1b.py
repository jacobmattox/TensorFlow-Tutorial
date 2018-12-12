from Week07Tuesdaya import *
from tensorflow.examples.tutorials.mnist import input_data


def main():


    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 28, 28, 1])
    filterTensor = tf.placeholder(tf.float32, [2, 2, 1, 1])

    myFilterHorizontal = [[.5, -.5],
                [.5, -.5]]

    myFilterHorizontal = np.array(myFilterHorizontal).reshape((2, 2, 1, 1))

    myFilterVertical = [[-.5, -.5],
                        [.5, .5]]
    myFilterVertical = np.array(myFilterVertical).reshape((2, 2, 1, 1))

    layer = tf.nn.conv2d(input=xTensor,
                         filter=filterTensor,
                         strides=[1, 1, 1, 1],
                         padding="VALID")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        resultHorizontal = sess.run(layer, feed_dict={xTensor:Xtrain[0:5].reshape(5, 28, 28, 1),
                                            filterTensor: myFilterHorizontal})
        resultVertical = sess.run(layer, feed_dict={xTensor: Xtrain[0:5].reshape(5, 28, 28, 1),
                                           filterTensor: myFilterVertical})


        im1before = Xtrain[0].reshape(28, 28)
        im1after = np.abs(resultHorizontal[0].reshape(27, 27))
        im2after = np.abs(resultVertical[0].reshape(27, 27))

        f, axarr = plt.subplots(1, 3)

        axarr[0].imshow(im1before, cmap='gray')
        axarr[1].imshow(im1after, cmap='gray')
        axarr[2].imshow(im2after, cmap='gray')

        f.show()


main()

