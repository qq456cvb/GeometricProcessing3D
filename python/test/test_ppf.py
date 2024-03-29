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
    
    ppf = pygeom.PPF(0.1, 12. / 180. * np.pi, 0.1, 20. / 180. * np.pi, 5)
    ppf.setup_model(model_pc)
    poses = ppf.detect(scene_pc)
    for p in poses[:10]:
        print(p.vote)
        print(p.r)
        print(p.t)