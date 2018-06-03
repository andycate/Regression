import numpy as np
import data_aggregation as da
import loss_function as lf
import math

training_images, training_labels = da.load_training_data()
theta = np.random.rand(785) / 100
learning_rate = 0.00003
iterations = 10000

for i in range(iterations):
    print(lf.j_of_theta(theta, training_images, training_labels))
    gradient = lf.gradient_j_of_theta(theta, training_images, training_labels).reshape(-1)
    theta = theta - (gradient * learning_rate)