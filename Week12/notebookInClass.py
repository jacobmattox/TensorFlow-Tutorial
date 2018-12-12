#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import math
from random import *
import boto3
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# In[28]:


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, seed=123)

Xtrain, Ytrain = mnist.train.next_batch(50000)
Xtest, Ytest = mnist.test.next_batch(10000)

xTensor = tf.placeholder(tf.float32, [None, 784])
yTensor = tf.placeholder(tf.float32, [None, 10])

wHiddenLayer1 = tf.Variable(tf.random_normal([784, 512]))
bHiddenLayer1 = tf.Variable(tf.random_normal([1, 512]))

outputOfFirst = tf.matmul(xTensor, wHiddenLayer1) + bHiddenLayer1

# Trying sigmoid function first
outputOfFirst = tf.nn.sigmoid(outputOfFirst)

wOutputLayer = tf.Variable(tf.random_normal([512, 10]))
bOutputLayer = tf.Variable(tf.random_normal([1, 10]))

yHat = tf.matmul(outputOfFirst, wOutputLayer) + bOutputLayer

yHatSoftmax = tf.nn.softmax(yHat)

lastError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yHat, labels=yTensor))

optimizer = tf.train.GradientDescentOptimizer(.3).minimize(lastError)

bucket_name = "elasticbeanstalk-us-east-1-536497194810"
s3 = boto3.resource("s3")

init = tf.global_variables_initializer()

numOfEpochs = 2000

with tf.Session() as sess:
        sess.run(init)

        allIndicesOfTraining = list(range(len(Xtrain)))
        shuffle(allIndicesOfTraining)
        
        #[1, 3, 5]
        #[213, 234, 234, 2,3 41,23 12,3 123, 13, 54]
        
        currBatchXtrain = np.take(Xtrain, allIndicesOfTraining[:2000], axis=0)
        currBatchYtrain = np.take(Ytrain, allIndicesOfTraining[:2000], axis=0)
        
        for el in range(numOfEpochs):
            sess.run(optimizer, feed_dict={xTensor: currBatchXtrain,
                                           yTensor: currBatchYtrain})
            if (el + 1) % 20 == 0:
                file_name = "newDirectory/TF/" + str(el) + ".txt"
                s3.Bucket(bucket_name).put_object(Key=file_name, Body="".encode("utf-8"))
                
            print(el)
        
        resultTF = sess.run(yHatSoftmax, feed_dict={xTensor: Xtest})

        indexOfHighest = np.argmax(resultTF, axis=1)
        indexOfActual = np.argmax(Ytest, axis=1)

        numOfCorrect = 0
        for idx, el in enumerate(indexOfHighest):

            if el == indexOfActual[idx]:
                numOfCorrect = numOfCorrect + 1
        
        print(numOfCorrect)
        file_name = "newDirectory/TF/result.txt"
        s3.Bucket(bucket_name).put_object(Key=file_name, Body=str(numOfCorrect).encode("utf-8"))
        
        
            

