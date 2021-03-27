import sys
import os
sys.path.append('build')
import pygeom
import numpy as np


if __name__ == '__main__':
    a = np.ones((10, 3), dtype=np.int32)
    print(a.shape)
    a[:2] = 3
    vec = pygeom.VectorXYZi(a)
    b = np.array(vec, dtype=np.int32)
    print(vec)
    print(b)