#include "geometry/pointcloud.h"


PointCloud::PointCloud()
{
}

PointCloud::~PointCloud()
{
}

PointCloud::PointCloud(std::vector<xyz> &&v)
    : verts(std::move(v)) {

}


PointCloud::PointCloud(std::vector<xyz> &&v, std::vector<xyz> &&n)
    : verts(std::move(v)), normals(std::move(n)) {

}