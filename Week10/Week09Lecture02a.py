from Week07Tuesdaya import *
import numpy as np
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
    plt.savefig("handwritingConfustionMatrixUsingNearestNeighbor")
    plt.show()

def main():

    timeStart = time()
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

    Xtrain, Ytrain = mnist.train.next_batch(60000)
    Xtest, Ytest = mnist.test.next_batch(500)

    myLabels = np.zeros((len(Xtest), ), dtype=np.uint8)
    myPredictions = np.zeros((len(Xtest), ), dtype=np.uint8)
    correct = 0
    for el in range(len(Xtest)):
        diff = np.sum(abs(Xtrain - Xtest[el]), 1)
        indexOfBestImage = np.argmin(diff)

        myLabels[el] = np.argmax(Ytest[el])
        myPredictions[el] = np.argmax(Ytrain[indexOfBestImage])
        if myLabels[el] == myPredictions[el]:
            correct += 1
    drawConfusion(_myLabels=myLabels,
                  _myPredictions=myPredictions)

    timeStop = time()
    print(timeStop - timeStart)
    print(correct)

    # 10k training
    # 464
    # 5.88 seconds
    # 60k training
    # 73.78100299835205
    # 478
main()