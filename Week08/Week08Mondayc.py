from Week07Tuesdaya import *

def main():

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    Xtrain, Ytrain = mnist.train.next_batch(60000)
    Xtest, Ytest = mnist.test.next_batch(10000)
    count = 0

    for i in range(len(Xtest)):
        closestDistance = 100000
        furthestDistance = 0
        indexOfBestImage = -1
        indexOfWorstImage = -1

        diff = np.sum(abs(Xtrain - Xtest[i]), 1)
        bestIndex = np.argmin(diff)
        #
        # for idx, el in enumerate(Xtrain):
        #     oldDiff = sum(abs(el - Xtest[i]))
        #     if oldDiff < closestDistance:
        #         closestDistance = oldDiff
        #         indexOfBestImage = idx
        #     if oldDiff > furthestDistance:
        #         furthestDistance = oldDiff
        #         indexOfWorstImage = idx

        #
        # f, axarr = plt.subplots(1,3)
        #
        # myImage = Xtest[i].reshape(28, 28) * 255
        # myImage = myImage.astype(np.uint8)
        # axarr[0].imshow(myImage, cmap='gray')
        #
        # myImage2 = Xtrain[indexOfBestImage].reshape(28, 28) * 255
        # myImage2 = myImage2.astype(np.uint8)
        # axarr[1].imshow(myImage2, cmap='gray')
        #
        # myImage3 = Xtrain[indexOfWorstImage].reshape(28, 28) * 255
        # myImage3 = myImage3.astype(np.uint8)
        # axarr[2].imshow(myImage3, cmap='gray')
        #
        # f.show()
        if Ytrain[bestIndex].tolist().index(1) == Ytest[i].tolist().index(1):
            count += 1

    print(count)

main()