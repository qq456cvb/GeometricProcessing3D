# GeometricProcessing3D

A small C++/CUDA 3D geometry processing library whose centerpiece is a **GPU implementation of Point Pair Feature (PPF) matching** for 6D object pose estimation, following [Drost et al., *Model Globally, Match Locally: Efficient and Robust 3D Object Recognition* (CVPR 2010)](https://ieeexplore.ieee.org/document/5540108), with refinements from [Vidal et al. (Sensors 2018)](https://www.mdpi.com/1424-8220/18/8/2678). Comes with Python bindings (`pygeom`).

## What's Inside

- **`src/algorithm/ppf.cu`** — the PPF detector, implemented end-to-end on the GPU with Thrust:
  - *Offline*: all model point-pair features (distance + three angles, quantized by `dist_delta`/`angle_delta`) are hashed into a sorted GPU table together with each pair's alignment transform.
  - *Online*: scene reference points compute their PPFs, look up matching model pairs, and vote in Hough accumulators over (model point, planar rotation angle); peaks become pose hypotheses.
  - *Post-processing*: poses are clustered by translation/rotation thresholds (`cluster_dist_th`, `cluster_angle_th`), votes are pooled, and cluster poses are averaged with proper quaternion averaging before returning a vote-sorted list of `Pose` (R, t, vote count).
- **`include/geometry`, `src/geometry`** — `PointCloud` and a `TriangleMesh` with adjacency structures and geodesic distance computation.
- **`src/io/objreader.cpp`** — minimal OBJ mesh and PCD point-cloud loading (sample bunny/chair data in `examples/data`).
- **`pybind/`** — the `pygeom` Python module exposing `PointCloud`, `PPF.setup_model`, and `PPF.detect`.

## Building

Requires CMake ≥ 3.14, a CUDA toolkit, Eigen3, and pybind11 (Python ≥ 3.6). The default architecture flag is `compute_75` (Turing); adjust `CMakeLists.txt` for your GPU.

```bash
mkdir build && cd build
cmake .. && cmake --build . --config Release
```

This produces the `gp3d` library, a `main` example (detects the chair model in a rotated scene), and the `pygeom` module.

## Python Usage

```python
import pygeom
import numpy as np

model = pygeom.PointCloud(pygeom.VectorXYZf(points), pygeom.VectorXYZf(normals))
ppf = pygeom.PPF(0.1, 12 / 180 * np.pi, 0.1, 20 / 180 * np.pi, 5)
ppf.setup_model(model)
poses = ppf.detect(scene)   # list of poses with .r, .t, .vote
```

See `python/test/test_ppf.py` for a runnable example.

## References

```bibtex
@inproceedings{drost2010model,
  title={Model Globally, Match Locally: Efficient and Robust 3D Object Recognition},
  author={Drost, Bertram and Ulrich, Markus and Navab, Nassir and Ilic, Slobodan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2010}
}

@article{vidal2018method,
  title={A Method for 6D Pose Estimation of Free-Form Rigid Objects Using Point Pair Features on Range Data},
  author={Vidal, Joel and Lin, Chyi-Yeu and Llad{\'o}s, Xavier and Mart{\'i}, Robert},
  journal={Sensors},
  volume={18},
  number={8},
  year={2018}
}
```
