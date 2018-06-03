import numpy as np
import math
import data_aggregation as da
import loss_function as lf
import checking

training_images, training_labels = da.load_training_data()
theta = np.random.rand(785) / 10000
learning_rate = 0.00003
iterations = 200
batch_size = 1000

testing_images, testing_labels = da.load_test_data()
print(checking.percent_accuracy(theta, testing_images, testing_labels))

for i in range(iterations):
    batch_imgs, batch_lbls = da.batch(training_images, training_labels, batch_size)
    print(lf.j_of_theta(theta, batch_imgs, batch_lbls))
    gradient = lf.gradient_j_of_theta(theta, batch_imgs, batch_lbls).reshape(-1)
    theta = theta - (gradient * learning_rate)

testing_images, testing_labels = da.load_test_data()
print(checking.percent_accuracy(theta, testing_images, testing_labels))