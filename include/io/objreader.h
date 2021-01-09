#ifndef OBJREADER_H
#define OBJREADER_H

#include <armadillo>
#include "geometry/trianglemesh.h"

class ObjReader
{
private:
    
public:
    ObjReader();
    ~ObjReader();
    std::shared_ptr<TriangleMesh> read_mesh(const char *fn);
};

#endif