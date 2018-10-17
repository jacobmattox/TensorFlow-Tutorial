from Week07Tuesdaya import *

def main():

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    Xtrain, Ytrain = mnist.train.next_batch(60000)
    Xtest, Ytest = mnist.test.next_batch(10000)
    Xtrain = Xtrain.astype(np.float16)
    Xtest = Xtest.astype(np.float16)

    closestDistance = 100000
    furthestDistance = 0
    indexOfBestImage = -1
    indexOfWorstImage = -1
    print("Shape of first image is", Xtest[0].shape)
    print("Shape of all training is", Xtrain.shape)
    #
    # newVectorXtrain = Xtrain.reshape(1, Xtrain.shape[0] * Xtrain.shape[1])
    # print(newVectorXtrain.shape)
    #
    # newXtest = np.repeat(Xtest, 10000, 0)
    # print(newXtest.shape)
    #
    # testMat = np.array([[1, 2, 3],
    #                     [4, 5, 6]])
    # print(np.repeat(testMat, 10, 0))
    #
    # testCube = np.array([[[1, 2], [2, 4]], [[5, 6], [7, 8]]])
    # print(testCube.shape)
    # d = np.sum(testCube, 2)
    # print(d)

    allBest = []
    allWorst = []

    for testInstance in Xtest:
        diff = np.sum(abs(Xtrain - testInstance), 1)
        allBest.append(np.argmin(diff))
        allWorst.append(np.argmax(diff))

    f, axarr = plt.subplots(1,3)

    myImage = Xtest[10].reshape(28, 28) * 255
    myImage = myImage.astype(np.uint8)
    axarr[0].imshow(myImage, cmap='gray')

    myImage2 = Xtrain[allBest[10]].reshape(28, 28) * 255
    myImage2 = myImage2.astype(np.uint8)
    axarr[1].imshow(myImage2, cmap='gray')

    myImage3 = Xtrain[allWorst[10]].reshape(28, 28) * 255
    myImage3 = myImage3.astype(np.uint8)
    axarr[2].imshow(myImage3, cmap='gray')

    f.show()
    print(Ytrain[allBest[10]])
    print(Ytest[10])

main()