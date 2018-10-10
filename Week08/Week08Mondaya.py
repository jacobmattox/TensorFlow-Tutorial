from Week07Tuesdaya import *
from sklearn import *
import tensorflow as tf
import time
import numpy as np

myArray1 = np.random.randint(0, 10, (1, 4))
myArray2 = np.random.randint(0, 10, (4, 4))

print(myArray1, myArray1.shape)
print(myArray2, myArray2.shape)

additionResult = myArray1 + myArray2
print(additionResult)

#returns index of the max in the column (change to 1 for row)
print(np.argmax(additionResult, 0))