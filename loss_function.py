import numpy as np
import math

def h_of_x(params, x):
    return 1 / (np.exp(-params.dot(x)) + 1)

def j_of_theta(params, x, labels): # parameters should be of length 785(for this algorithm)
    y_hat = h_of_x(params, x)
    result = labels*np.log(y_hat) + (1-labels)*np.log(1-y_hat)
    return -np.sum(result)