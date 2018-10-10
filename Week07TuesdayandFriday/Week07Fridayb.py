
from Week07Tuesdaya import *
from sklearn import *
import tensorflow as tf
import time


def calcTotalError(_W, _X, _b, _Y):
    myResult = 0
    for idx, el in enumerate(_X):
        myResult = myResult + (_Y[idx] - calcYHat(_W, el, _b)) ** 2

    return myResult/len(_X)


def calcYHat(_W, _X, _b):
    return np.sum(_W * _X.transpose()) + _b

def main():

    allX, allY = readFile([5, 14], [2])


    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w1 = np.random.randn()
    w2 = np.random.randn()
    bB = np.random.randn()
    W = tf.Variable([[w1],
                     [w2]], name="weights")
    b = tf.Variable(bB, name="bias")

    Y_hat = tf.matmul(X, W) + b

    n = len(allX)

    myLossFunction = tf.reduce_sum(tf.pow(Y_hat - Y, 2)) / n

    optimizer = tf.train.GradientDescentOptimizer(.1).minimize(myLossFunction)

    init = tf.global_variables_initializer()

    # plt.plot(allX, allY, 'ro', label="Original data")
    # plt.legend()
    # plt.plot(allX, allX * w1 + bB, 'b', label="Predicted values")
    with tf.Session() as sess:
        sess.run(init)

        for el in range(10000):
            sess.run(optimizer, feed_dict={X: allX, Y: allY})
            print("W:", sess.run(W))
            print("b:", sess.run(b))

        #
        # plt.plot(allX, allY, 'ro', label="Original data")
        # plt.plot(allX, sess.run(W) * allX + sess.run(b), 'b', label="Predicted values")
        # plt.legend()
        # plt.show()


main()
