import numpy as np
import math

"""compute the logistic function"""
def h_of_x(theta, x):
    dot_product = -theta.dot(x) # compute the linear function of the images with parameter theta
    return 1 / (np.exp(dot_product) + 1) # apply the logistic function to restrict the range to (0, 1)

"""compute the loss of the model"""
def j_of_theta(theta, x, labels):
    y_hat = h_of_x(theta, x) # compute the model's predictions
    result = labels*np.log(y_hat) + (1-labels)*np.log(1-y_hat) # compute individual negative losses across all images
    return -np.sum(result) # sum together the resulting positive loss

"""compute the gradient of the loss"""
def gradient_j_of_theta(theta, x, labels):
    return x.dot((h_of_x(theta, x) - labels).reshape(-1, 1)) # simple gradient computation across image batch