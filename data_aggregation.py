import numpy as np
import struct

def format_images(data_file):
    # read data into numpy array
    images = np.array([])
    try:
        magic_num = struct.unpack(">L", data_file.read(4))[0]
        num_images = struct.unpack(">L", data_file.read(4))[0]
        rows = struct.unpack(">L", data_file.read(4))[0] # per image
        cols = struct.unpack(">L", data_file.read(4))[0] # per image

        img_buffer = data_file.read(num_images * rows * cols) # reads all the data for all the images
        dt = np.dtype(np.uint8).newbyteorder('>')
        img_array = np.frombuffer(img_buffer, dtype=dt, count=-1, offset=0)
        # reshape array
        img_array = np.reshape(img_array, (num_images, rows * cols)).transpose()
        # change data type to float32
        img_array = img_array.astype(dtype=np.float32, casting='safe')
        # append ones for intercept term
        img_array = np.append(img_array, np.ones((1, num_images)), axis=0)
        images = img_array
    finally:
        return images

def format_labels(data_file):
    # read data into array
    # change data type to float32

def process_data(imgs, lbls):
    # select all ones and all zeros
    # append ones to image data for intercept term

def load_training_data():
    # open the raw data files
    training_images_raw = open("train-images-idx3-ubyte", "rb")
    training_labels_raw = open("train-labels-idx1-ubyte", "rb")

    # create numpy arrays with the correct (raw) data
    training_images = format_images(training_images_raw)
    training_labels = format_labels(training_labels_raw)

    # close input streams
    training_images_raw.close()
    training_labels_raw.close()

    # process the data, and prepare it for training
    training_images, training_labels = process_data(training_images, training_labels)

    # return the processed data
    return training_images, training_labels

def load_test_data():
    # open the raw data files
    test_images_raw = open("t10k-images-idx3-ubyte", "rb")
    test_labels_raw = open("t10k-labels-idx1-ubyte", "rb")

    # create numpy arrays with the correct (raw) data
    test_images = format_images(test_images_raw)
    test_labels = format_labels(test_labels_raw)

    # close input streams
    test_images_raw.close()
    test_labels_raw.close()

    # process the data, and prepare it for training
    test_images, test_labels = process_data(test_images, test_labels)

    # return the processed data
    return test_images, test_labels