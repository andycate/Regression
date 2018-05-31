import numpy as np
import struct

f = open("train-images-idx3-ubyte", "rb")
try:
    byte = f.read(4)
    byte = f.read(4)
    byte = f.read(4)
    byte = f.read(4)
    byte = f.read(1)
    for y in range(28):
        for x in range(28):
            bit = str(len(str(struct.unpack(">b", byte)[0])))
            print(bit+bit, end="", flush=True)
            byte = f.read(1)
        print("")
finally:
    f.close()