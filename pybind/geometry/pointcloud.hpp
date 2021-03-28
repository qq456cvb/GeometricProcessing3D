#ifndef POINTCLOUD_HPP
#define POINTCLOUD_HPP

#include <pybind11/pybind11.h>
#include "geometry/pointcloud.h"

void pybind_pointcloud(pybind11::module &m) {
    pybind11::class_<PointCloud, std::shared_ptr<PointCloud>>(m, "PointCloud")
        .def(pybind11::init())
        .def(pybind11::init<const std::vector<xyz> &, const std::vector<xyz> &>())
        .def_readwrite("points", &PointCloud::verts)
        .def_readwrite("normals", &PointCloud::normals);
}

#endif