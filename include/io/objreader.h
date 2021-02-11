#ifndef OBJREADER_H
#define OBJREADER_H

#include <armadillo>
#include "geometry/trianglemesh.h"
#include "geometry/pointcloud.h"

class ObjReader
{
private:
    
public:
    ObjReader();
    ~ObjReader();
    std::shared_ptr<TriangleMesh> read_mesh(const char *fn);
    std::shared_ptr<PointCloud> read_cloud(const char *fn);
};

#endif