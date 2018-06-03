import numpy as np
import math
import data_aggregation as da
import loss_function as lf
import checking

training_images, training_labels = da.load_training_data()
theta = np.random.rand(785) / 10000
learning_rate = 0.000025
iterations = 800

testing_images, testing_labels = da.load_test_data()
print(checking.percent_accuracy(theta, testing_images, testing_labels))

for i in range(iterations):
    print(lf.j_of_theta(theta, training_images, training_labels))
    gradient = lf.gradient_j_of_theta(theta, training_images, training_labels).reshape(-1)
    theta = theta - (gradient * learning_rate)

testing_images, testing_labels = da.load_test_data()
print(checking.percent_accuracy(theta, testing_images, testing_labels))