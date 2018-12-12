
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.metrics import confusion_matrix

#reads file listed for these columns and returns those as numpy arrays
def readFile(_inputColNums, _outputColNums):
    inputData = pd.read_csv("kc_house_data.csv", sep = ',', usecols = _inputColNums,
                            header = 0)
    outputData = pd.read_csv("kc_house_data.csv", sep = ',', usecols = _outputColNums,
                            header = 0)

    tempAllX = np.array(inputData, dtype="float")

    avgOfX1 = np.average(tempAllX[:, 0])
    avgOfX2 = np.average(tempAllX[:, 1])

    tempAllX[:, 0] = (tempAllX[:, 0] - avgOfX1) /(np.max(tempAllX[:, 0]) - np.min(tempAllX[:,0]))
    tempAllX[:, 1] = (tempAllX[:, 1] - avgOfX2) /(np.max(tempAllX[:, 1]) - np.min(tempAllX[:,1]))

    return tempAllX, np.array(outputData, dtype='float64')

