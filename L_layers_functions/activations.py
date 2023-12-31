import numpy as np
import h5py
import matplotlib.pyplot as plt


def sigmoid(W,A,b):

    z =  np.dot(W,A)+b


    s=1/(1+np.exp(-z))

    return s , z


def sigmoid_backward (dA , z):

    dZ = dA*(np.exp(-z)/((1+np.exp(-z))**2))

    return dZ


def relu (W,A,b):

    z =  np.dot(W,A)+b

    s = np.max(0,z)

    return s , z


def relu_backward (dA , z):

    if z<0 :
            dZ = 0
    else:
         dZ = dA

    return dZ
