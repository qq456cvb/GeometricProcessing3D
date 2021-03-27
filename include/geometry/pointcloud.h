#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

using xyz = Eigen::Vector3f;

class PointCloud
{
private:
public:
    std::vector<xyz> verts;
    std::vector<xyz> normals;
    PointCloud();
    PointCloud(const PointCloud &pc);
    ~PointCloud();
    PointCloud(std::vector<xyz> &&v);
    PointCloud(std::vector<xyz> &&v, std::vector<xyz> &&n);
    PointCloud(const std::vector<xyz> &v, const std::vector<xyz> &n);
};

#endif