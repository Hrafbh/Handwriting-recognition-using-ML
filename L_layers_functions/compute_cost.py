import numpy as np
import h5py
import matplotlib.pyplot as plt
from activations import sigmoid, sigmoid_backward, relu, relu_backward


np.random.seed(1)

def compute_cost(AL, Y):

    
    m = Y.shape[1]

    # Compute loss from aL and y.

    cost = (-1/m)*(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    
    return cost