import numpy as np
import math

def h_of_x(theta, x):
    dot_product = -theta.dot(x)
    return 1 / (np.exp(dot_product) + 1)

def j_of_theta(theta, x, labels): # parameters should be of length 785(for this algorithm)
    y_hat = h_of_x(theta, x)
    result = labels*np.log(y_hat) + (1-labels)*np.log(1-y_hat)
    return -np.sum(result)

def gradient_j_of_theta(theta, x, labels):
    return x.dot((h_of_x(theta, x) - labels).reshape(-1, 1))