import numpy as np
import struct

training_images = open("train-images-idx3-ubyte", "rb")

try:
    magic_num = struct.unpack(">L", training_images.read(4))[0]
    num_images = struct.unpack(">L", training_images.read(4))[0]
    rows = struct.unpack(">L", training_images.read(4))[0] # per image
    cols = struct.unpack(">L", training_images.read(4))[0] # per image
    img_buffer = training_images.read(num_images*rows*cols) # reads all the data for all the images
    print(len(img_buffer))
    dt = np.dtype(np.uint8)
    dt = dt.newbyteorder('>')
    img_array = np.frombuffer(img_buffer, dtype=dt, count=-1, offset=0)
    print(img_array.size)
finally:
    training_images.close()