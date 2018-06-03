import numpy as np
import math
import loss_function as lf

"""calculate the percent accuracy of the model, based on how many images it classifies correctly"""
def percent_accuracy(theta, x, labels):
    predictions = lf.h_of_x(theta, x) # call the logistic function with the model parameters and images
    delta = np.abs(labels - predictions) # find the difference between the labels and the predictions
    return float(np.where(delta < 0.5)[0].shape[0]) / float(predictions.shape[0]) # if the difference is less than 0.5, the claccificaation is correct

"""perform gradient checking"""
def gradient_checking(theta, x, labels, epsilon=0.0001):
    params = np.identity(theta.shape[-1], dtype=np.float32) * epsilon # gemerate the matrix of epsilons
    theta = np.repeat(theta.reshape(1, -1), theta.shape[-1], axis=0) # reshape the parameters so that each set has a slight variation(epsilon)
    return (lf.j_of_theta(theta + params, x, labels) - lf.j_of_theta(theta - params, x, labels)) / (2 * epsilon) # calculate the approx gradient