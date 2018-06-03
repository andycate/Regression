import numpy as np
import math
import data_aggregation as da
import loss_function as lf
import checking

training_images, training_labels = da.load_training_data() # load the training images and labels
theta = np.random.rand(785) / 10000 # initialize the parameters to very small random values
learning_rate = 0.00003 # set the learning rate
iterations = 200 # set the number of iterations
batch_size = 1000 # set the batch size (-1 uses the entire training set size)

test_images, test_labels = da.load_test_data() # load the test images and labels
print(checking.percent_accuracy(theta, test_images, test_labels)) # print out the percent accuracy of the model before training

for i in range(iterations): # train the model
    batch_imgs, batch_lbls = da.batch(training_images, training_labels, batch_size) # get a new random batch
    # print(lf.j_of_theta(theta, batch_imgs, batch_lbls)) # print out the raw loss from the loss function
    gradient = lf.gradient_j_of_theta(theta, batch_imgs, batch_lbls).reshape(-1) # calculate the gradient of the loss
    print(np.average(gradient - checking.gradient_checking(theta, batch_imgs, batch_lbls))) # print out the difference between the gradient and the approx gradient
    theta = theta - (gradient * learning_rate) # apply the gradient to the model parameters

print(checking.percent_accuracy(theta, test_images, test_labels)) # test the percent accuracy of the model after training