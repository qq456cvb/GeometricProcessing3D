#ifndef OBJREADER_H
#define OBJREADER_H

#include <armadillo>

using face_idx_t = std::tuple<int, int, int>;
class ObjReader
{
private:
    std::vector<float> verts;
    std::vector<face_idx_t> face_idxs;
public:
    ObjReader(const char *fn);
    ~ObjReader();
};

#endif