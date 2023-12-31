import numpy as np
import h5py
import matplotlib.pyplot as plt

from L_layers_functions.initialize_parameters import initialize_parameters_deep
from L_layers_functions.L_model_forward import L_model_forward
from L_layers_functions.compute_cost import compute_cost
from L_layers_functions.L_model_backward import L_model_backward
from L_layers_functions.update_parameters import update_parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):


    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        # Compute cost.

        cost = compute_cost(AL, Y)

        # Backward propagation.

        grads = L_model_backward(AL, Y, caches)

        # Update parameters.

        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs