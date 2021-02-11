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