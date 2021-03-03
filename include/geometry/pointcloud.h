#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <memory>
#include <vector>

using xyz = std::array<float, 3>;

class PointCloud
{
private:
    std::vector<xyz> verts;
    std::vector<xyz> normals;
public:
    PointCloud();
    ~PointCloud();
    PointCloud(std::vector<xyz> &&v);
    PointCloud(std::vector<xyz> &&v, std::vector<xyz> &&n);
};

#endif