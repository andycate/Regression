import numpy as np
import math
import loss_function as lf

"""returns the percent accuracy of the model, based on how many images it classifies correctly"""
def percent_accuracy(theta, x, labels):
    predictions = lf.h_of_x(theta, x) # call the logistic function with the model parameters and images
    delta = np.abs(labels - predictions) # find the difference between the labels and the predictions
    return float(np.where(delta < 0.5)[0].shape[0]) / float(predictions.shape[0]) # if the difference is less than 0.5, the claccificaation is correct