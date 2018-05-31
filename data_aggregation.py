import numpy as np
import struct

def format_images(data_file):
    # read data into numpy array
    # reshape array
    # change data type to float32
    # append ones for intercept term

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

    # process the data, and prepare it for training
    training_images, training_labels = process_data(training_images, training_labels)

    # return the processed data
    return training_images, training_labels

    # append 
    training_images = np.append(training_images, np.ones((1, num_images)), axis=0)

    try:
        magic_num = struct.unpack(">L", training_images.read(4))[0]
        num_images = struct.unpack(">L", training_images.read(4))[0]
        rows = struct.unpack(">L", training_images.read(4))[0] # per image
        cols = struct.unpack(">L", training_images.read(4))[0] # per image
        print("Total number of training images: " + str(num_images))
        print("Rows per image: " + str(rows))
        print("Columns per image: " + str(cols))

        img_buffer = training_images.read(num_images * rows * cols) # reads all the data for all the images
        print("Bytes in the image byte buffer: " + str(len(img_buffer)))
        dt = np.dtype(np.uint8)
        dt = dt.newbyteorder('>')
        img_array = np.frombuffer(img_buffer, dtype=dt, count=-1, offset=0)
        print("Raw array size: " + str(img_array.size))

        img_array = np.reshape(img_array, (num_images, rows * cols))
        print("Shape of new ndarray: " + str(img_array.shape))
        print(img_array)

        img_array = img_array.transpose()
        print("Shape of transposed ndarray: " + str(img_array.shape))

        for y in range(28):
            for x in range(28):
                print(str(len(str(img_array[28 * y + x][1])) - 1) + " ", end="", flush=True)
            print("")
    finally:
        training_images.close()

def load_test_data():
    # open the raw data files
    test_images_raw = open("t10k-images-idx3-ubyte", "rb")
    test_labels_raw = open("t10k-labels-idx1-ubyte", "rb")

    # create numpy arrays with the correct (raw) data
    test_images = format_images(test_images_raw)
    test_labels = format_labels(test_labels_raw)

    # process the data, and prepare it for training
    test_images, test_labels = process_data(test_images, test_labels)

    # return the processed data
    return test_images, test_labels