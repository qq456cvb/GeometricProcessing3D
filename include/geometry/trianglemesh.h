#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include <vector>
#include <tuple>
#include <memory>
#include <armadillo>

typedef struct {
    std::array<int, 3> vs;
    std::array<int, 3> ns;
    std::array<int, 3> ts;
} face_idx_t;

using xyz = std::array<float, 3>;
class TriangleMesh
{
private:
    std::vector<xyz> verts;
    std::vector<face_idx_t> face_idxs;

public:
    TriangleMesh();
    TriangleMesh(const TriangleMesh &t);
    TriangleMesh(TriangleMesh &&t);
    TriangleMesh(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) = delete;  // no copy
    TriangleMesh(std::vector<xyz> &&v, std::vector<face_idx_t> &&f);
    ~TriangleMesh();

    std::vector<float> geodesic(const std::vector<int> &source);
};

#endif