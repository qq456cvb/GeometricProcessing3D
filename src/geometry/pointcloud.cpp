#include "geometry/pointcloud.h"


PointCloud::PointCloud()
{
}

PointCloud::PointCloud(const PointCloud &pc)
{
    this->verts = pc.verts;
    this->normals = pc.normals;
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

PointCloud::PointCloud(const std::vector<xyz> &v, const std::vector<xyz> &n)
    : verts(v), normals(n) {
    
}

PointCloud::PointCloud(const std::vector<arma::fvec3> &v, const std::vector<arma::fvec3> &n) {
    verts.resize(v.size());
    normals.resize(n.size());
    for (auto &e: verts) e = arma::fvec(v);
    for (auto &e: normals) e = n; 
}