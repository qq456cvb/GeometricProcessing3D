#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "algorithm/ppf.h"

using namespace pybind11::literals;

void pybind_algorithm(pybind11::module &m) {
    pybind11::class_<PPF, std::shared_ptr<PPF>>(m, "PPF")
        .def(pybind11::init<float, float, float, float, int>(), "Super fast PPF algorithm!", "dist_delta"_a, "angle_delta"_a, "cluster_dist_th"_a, "cluster_angle_th"_a, "min_vote_th"_a)
        .def("setup_model", &PPF::setup_model, "pc"_a)
        .def("detect", &PPF::detect, "pc"_a);

    pybind11::class_<Pose, std::shared_ptr<Pose>>(m, "Pose")
        .def(pybind11::init<>())
        .def(pybind11::init<const Pose &>(), "pose"_a)
        .def_readonly("vote", &Pose::vote)
        .def_property_readonly("r", [](Pose &s) {return pybind11::array({3, 3}, s.r);})
        .def_property_readonly("t", [](Pose &s) {return pybind11::array(3, s.t);});
}

#endif