#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include <vector>
#include <tuple>
#include <memory>
#include <Eigen/Dense>

typedef struct {
    std::array<int, 3> vs;
    std::array<int, 3> ns;
    std::array<int, 3> ts;
} face_idx_t;

typedef std::unordered_map<std::pair<int, int>, std::vector<int>, std::function<size_t(std::pair<int, int>)>, 
        std::function<bool(std::pair<int, int>, std::pair<int, int>)>> pair_map;

using xyz = Eigen::Vector3f;
class TriangleMesh
{
private:
    std::vector<xyz> verts;
    std::vector<face_idx_t> face_idxs;

    std::vector<std::vector<int>> v2v_nbrs;
    std::vector<float> areas;
    std::vector<xyz> face_ns;
    std::vector<std::vector<int>> v2f_nbrs;
    pair_map e2f_nbrs;

public:
    TriangleMesh();
    TriangleMesh(const TriangleMesh &t);
    TriangleMesh(TriangleMesh &&t);
    TriangleMesh(std::vector<xyz> &&v, std::vector<face_idx_t> &&f);
    ~TriangleMesh();

    std::vector<float> geodesic(const std::vector<int> &source);
};

#endif