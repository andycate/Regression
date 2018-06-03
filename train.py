import numpy as np
import data_aggregation as da
import loss_function as lf
import math

training_images, training_labels = da.load_training_data()
params = np.random.rand(785)
print(lf.j_of_theta(params, training_images, training_labels))