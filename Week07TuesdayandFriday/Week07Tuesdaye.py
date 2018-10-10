
from Week07Tuesdaya import *
#slow regression by calculating the error everytime
def calcTotalError(_W, _X, _b, _Y):
    myResult = 0
    for idx, el in enumerate(_X):
        myResult = myResult + (_Y[idx] - calcYHat(_W, el, _b)) ** 2

    return myResult/len(_X)


def calcYHat(_W, _X, _b):
    return np.sum(_W * _X.transpose()) + _b

def main():
    allX, allY = readFile([5, 14], [2])
    updatedAllX1 = []
    updatedAllX2 = []

    print(allX)

    for idx, currVal in enumerate(allX):

        currVal[0] = (currVal[0] - np.mean(allX[:, 0]))/(np.max(allX[:, 0]) - np.min(allX[:, 0]))
        currVal[1] = (currVal[1] - np.mean(allX[:, 1]))/(np.max(allX[:, 1]) - np.min(allX[:, 1]))
    print(allX)
    #trying to create 2 new numpy arrays that are a combination of original array allX but scaled, this code sucks
    # for idx, currVal in enumerate(allX):
    #     updatedAllX1.append((currVal[0] - np.mean(allX[:, 0]))/ (np.max(allX[:, 0] - np.min(allX[:, 0]))))
    #     updatedAllX2.append((currVal[1] - np.mean(allX[:, 1]))/ (np.max(allX[:, 1] - np.min(allX[:, 1]))))
    #
    # updatedAllX1 = np.array(updatedAllX1).reshape(21613, 1)
    # updatedAllX2 = np.array(updatedAllX2).reshape(21613, 1)
    #
    # print(updatedAllX1.shape)
    # print(updatedAllX2.shape)
    # # need to figure out a way to combine these two arrays back into one numpy array
    # allX = np.stack(updatedAllX1, updatedAllX2)
    # print("allX with new numbers between 1 and -1", str(allX))

    learningRate = .1

    W = np.array([[0.0],
                 [0.0]])
    b = 0

    for el in range(1000):
        derWithRespectToW1 = 0
        derWithRespectToW2 = 0
        derWithRespectToB = 0
        for idx, currElement in enumerate(allX):
            derWithRespectToW1 = derWithRespectToW1 + (calcYHat(W, currElement, b) - allY[idx]) * currElement[0]
            derWithRespectToW2 = derWithRespectToW2 + (calcYHat(W, currElement, b) - allY[idx]) * currElement[1]
            derWithRespectToB = derWithRespectToB + (calcYHat(W, currElement, b) - allY[idx])

        derWithRespectToW1 = (derWithRespectToW1 * 2) / len(allX)
        derWithRespectToW2 = (derWithRespectToW2 * 2) / len(allX)
        derWithRespectToB = (derWithRespectToB * 2) / len(allX)


        W[0][0] = W[0][0] - learningRate * derWithRespectToW1
        W[1][0] = W[1][0] - learningRate * derWithRespectToW2
        b = b - learningRate * derWithRespectToB

        if el % 10 == 0:
            print("b:", str(b))
            print("W[0][0]:", str(W[0][0]))
            print("W[1][0]:", str(W[1][0]))
            print("Total error is:", str(calcTotalError(W, allX, b, allY)))

main()