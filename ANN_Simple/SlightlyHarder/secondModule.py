# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:48:24 2017

@author: FreieL01
"""

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function 
def nonlin(x,deriv=False):
    if(deriv==True):
#        return x*(1-x)
        return np.divide(np.exp(x),pow(np.exp(x)+1,2))
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])


# output 
y = np.array([[0,1,1,0]]).T
           
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# Its dimension is (3,1) because we have 3 inputs and 1 output.
# Thus, we want to connect every node in l0 to every node in l1, which requires a matrix of dimensionality (3,1). 
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# Save l1 values for later investigation
nIter = 60000;
#l1Array = np.empty((nIter,4))
l2Array = np.empty((nIter,4))
l2Error = np.empty((nIter,4))
                         
for iter in range(nIter):
    
    # forward propagation
    l0 = X;
    l1 = nonlin(np.dot(l0,syn0)) 
    l2 = nonlin(np.dot(l1,syn1)) 
#    l1Array[iter,]  = l1.T;
    l2Array[iter,]  = l2.T;
    
   

    # how much did we miss
    l2_error = y - l2
    l2Error[iter,]  = l2_error.T;
    if (iter% 1E4) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
           
    # weight error by derivative
    l2_delta = l2_error * nonlin(l2,True)
        
    # Backpropagation
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1,True)
    
    # update weights
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)
    
print("Output After Training:")
print("L1: " +str(l1))
print("L2: " +str(l2))
