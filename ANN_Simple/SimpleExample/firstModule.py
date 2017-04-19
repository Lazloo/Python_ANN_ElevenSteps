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
y = np.array([[0,0,1,1]]).T
           
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

# Save l1 values for later investigation
nIter = 10000;
l1Array = np.empty((nIter,4))
l1Error = np.empty((nIter,4))
                         
for iter in range(nIter):
    
    # forward propagation
    l0 = X;
    #Weigthed sum is used as input for the sigmoid function --> 
    l1 = nonlin(np.dot(l0,syn0)) 
    l1Array[iter,]  = l1.T;
    
   
    
    # how much did we miss
    l1_error = y - l1
    l1Error[iter,]  = l1_error.T;
    
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    
    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    
print("Output After Training:")
print(l1)

plt.figure()
line1, = plt.plot(range(nIter), l1Array[:,0],'b')
line2, = plt.plot(range(nIter), l1Array[:,1],'k')
line3, = plt.plot(range(nIter), l1Array[:,2],'r')
line4, = plt.plot(range(nIter), l1Array[:,3],'y')
plt.legend([line1,line2,line3,line4], ['A','B','C','D'])
plt.title('Estimated Output')

plt.figure()
line1, = plt.plot(range(nIter), l1Error[:,0],'b')
line2, = plt.plot(range(nIter), l1Error[:,1],'k')
line3, = plt.plot(range(nIter), l1Error[:,2],'r')
line4, = plt.plot(range(nIter), l1Error[:,3],'y')
plt.legend([line1,line2,line3,line4], ['A','B','C','D'])
plt.title('Prediction Error')


# Final Weights are contra intuitive. Simple solution would be [1,0,0] but [9.7,-0.2,-4.6] --> Overfittig 
syn0
xTest = np.linspace(-5,5,100);
plt.figure()
plt.plot(xTest,nonlin(xTest))
