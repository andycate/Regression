import numpy as np
import math
import loss_function as lf

def percent_accuracy(theta, x, labels):
    predictions = lf.h_of_x(theta, x)
    delta = np.abs(labels - predictions)
    return float(np.where(delta < 0.5)[0].shape[0]) / float(predictions.shape[0])