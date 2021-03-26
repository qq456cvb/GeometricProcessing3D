#ifndef BIND_CPP
#define BIND_CPP

#include <armadillo>
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

PYBIND11_MAKE_OPAQUE(std::vector<arma::ivec2>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::fvec2>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::vec2>);

PYBIND11_MAKE_OPAQUE(std::vector<arma::ivec3>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::fvec3>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::vec3>);

PYBIND11_MAKE_OPAQUE(std::vector<arma::ivec4>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::fvec4>);
PYBIND11_MAKE_OPAQUE(std::vector<arma::vec4>);

#include "algorithm/algorithm.hpp"
#include "arma.hpp"


PYBIND11_MODULE(pygeom, m) {
    m.doc() = "Python binding of Geometric3D";
    pybind_algorithm(m);
    pybind_arma(m);
}
#endif