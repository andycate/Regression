import numpy as np
import data_aggregation as da
import loss_function as lf
import math

training_images, training_labels = da.load_training_data()
theta = np.random.rand(785)/100
print(lf.j_of_theta(theta, training_images, training_labels))