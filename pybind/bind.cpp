#ifndef BIND_CPP
#define BIND_CPP

#include <Eigen/Dense>
#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<int32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2f>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4f>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4d>);

#include "algorithm/algorithm.hpp"
#include "eigen.hpp"
#include "geometry/pointcloud.hpp"

PYBIND11_MODULE(pygeom, m) {
    m.doc() = "Python binding of Geometric3D";
    pybind_pointcloud(m);
    pybind_algorithm(m);
    pybind_eigen(m);
    
}
#endif