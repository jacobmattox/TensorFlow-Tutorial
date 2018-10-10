
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

    learningRate = .00000001

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