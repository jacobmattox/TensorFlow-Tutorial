from Week07Tuesdaya import *
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data


def drawConfusion(_myLabels, _myPredictions):
    returnedCM = confusion_matrix(_myLabels, _myPredictions)

    plt.imshow(returnedCM, cmap=plt.cm.Blues)
    plt.xticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))
    plt.yticks(np.arange(np.min(_myLabels), np.max(_myLabels) + 1))

    plt.xlabel("My Predicted")
    plt.ylabel("Labels")

    plt.show()

    print(returnedCM)


def main():

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    print(mnist)

    Xtrain, Ytrain = mnist.train.next_batch(5000)
    Xtest, Ytest = mnist.test.next_batch(2000)

    myLabels = np.zeros((len(Xtest), ), dtype=np.uint8)
    myPredictions = np.zeros((len(Xtest), ), dtype=np.uint8)


    for el in range(len(Xtest)):
        diff = np.sum(abs(Xtrain - Xtest[el]), 1)
        indexOfBestImage = np.argmin(diff)

        myLabels[el] = np.argmax(Ytest[el])
        myPredictions[el] = np.argmax(Ytrain[indexOfBestImage])


    print(myPredictions)
    print(myLabels)

    drawConfusion(_myLabels=myLabels,
                  _myPredictions=myPredictions)

    return

main()