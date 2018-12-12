from Week07Tuesdaya import *
from tensorflow.examples.tutorials.mnist import input_data


def main():


    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(50000)
    Xtest, Ytest = mnist.test.next_batch(500)

    xTensor = tf.placeholder(tf.float32, [None, 28, 28, 1])
    filterTensor = tf.placeholder(tf.float32, [3, 3, 1, 1])

    myFilter = [[.05, .1, .05],
                [.10, .4, .10],
                [.05, .1, .05]]

    myFilter = np.array(myFilter).reshape((3, 3, 1, 1))

    layer = tf.nn.conv2d(input=xTensor,
                         filter=filterTensor,
                         strides=[1, 1, 1, 1],
                         padding="VALID")

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        result = sess.run(layer, feed_dict={xTensor:Xtrain[0:5].reshape(5, 28, 28, 1),
                                            filterTensor: myFilter})


        im1before = Xtrain[0].reshape(28, 28)
        im1after = result[0].reshape(26, 26)

        f, axarr = plt.subplots(1, 2)

        axarr[0].imshow(im1before, cmap='gray')
        axarr[1].imshow(im1after, cmap='gray')



main()

