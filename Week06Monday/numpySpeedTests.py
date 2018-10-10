
import numpy as np
import time

myArray = np.array([1, 2, 3, 4])
print(myArray)
print(myArray.shape)

myArray = myArray.reshape(4, 1)
print(myArray.shape)

myArray = myArray + 5
myOtherArray = myArray + 1

myThirdArray = myArray * myOtherArray

print(myArray)
print(myOtherArray)
print(myThirdArray)

#creates a random numpy matrix with a million size
a = np.random.randint(1, 10, 1000000)
b = np.random.randint(1, 10, 1000000)


tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(toc-tic)

tic = time.time()
aTranspose = a.transpose()
cOther = np.matmul(aTranspose, b)
toc = time.time()
print(toc-tic)

tic = time.time()
mySum = 0
for el in range(1000000):
    mySum = mySum + a[el] * b[el]
toc = time.time()
print(mySum, toc-tic)
print(cOther)
print(c)

print(a)
print(b)

