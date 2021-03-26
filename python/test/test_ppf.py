import sys
import os
sys.path.append('build')
import pygeom
import numpy as np


if __name__ == '__main__':
    a = np.random.randn(1000, 3).astype(np.float32)
    b = np.random.randn(1000, 3).astype(np.float32)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    model_pc = pygeom.PointCloud(pygeom.VectorXYZf(a), pygeom.VectorXYZf(b))
    scene_pc = pygeom.PointCloud(pygeom.VectorXYZf(a.copy()), pygeom.VectorXYZf(b.copy()))
    
    ppf = pygeom.PPF(0.01, 12. / 180. * np.pi, 0.1, 20. / 180. * np.pi, 5)
    ppf.setup_model(model_pc)
    ppf.detect(scene_pc)