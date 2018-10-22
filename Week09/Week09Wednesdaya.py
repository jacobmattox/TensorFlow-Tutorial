
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():
    np.random.seed(0)
    myLabels = np.random.randint(0, 3, (9,))
    myPredictions = np.random.randint(0, 3, (9,))
    print(myPredictions)
    print(myLabels)

    returnedCM = confusion_matrix(myLabels, myPredictions)

    plt.imshow(returnedCM, cmap=plt.cm.Blues)

    plt.xticks(np.arange(3))
    plt.yticks([0, 1, 2])

    plt.xlabel("My predicted")
    plt.ylabel("Labels")

    plt.show()

    print(returnedCM)

main()