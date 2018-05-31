import numpy as np
import struct

def load_data():
    training_images = open("train-images-idx3-ubyte", "rb")
    training_labels_raw = open("train-labels-idx1-ubyte", "rb")
    test_images_raw = open("t10k-images-idx3-ubyte", "rb")
    test_labels_raw = open("t10k-labels-idx1-ubyte", "rb")

    # create numpy arrays with the correct data
    training_images = process_training_images(training_images_raw)
    training_labels = process_training_labels(training_labels_raw)
    test_images = process_test_images(test_images_raw)
    test_labels = process_test_labels(test_labels_raw)

    # append 
`   `

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