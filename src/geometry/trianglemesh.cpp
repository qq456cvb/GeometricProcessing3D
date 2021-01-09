#include "geometry/trianglemesh.h"


arma::sp_fmat build_adjs(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    static arma::sp_fmat adjs;  // N x N, 0, 1 adjacency
    if (adjs.empty()) {
        adjs.set_size(v.size(), v.size());
        for (auto &facet : f) {
            const auto &vs = facet.vs;
            adjs(vs[0], vs[1]) = 1.f;
            adjs(vs[1], vs[0]) = 1.f;
            adjs(vs[0], vs[2]) = 1.f;
            adjs(vs[2], vs[0]) = 1.f;
            adjs(vs[1], vs[2]) = 1.f;
            adjs(vs[2], vs[1]) = 1.f;
        }
    }
    return adjs;
}

arma::fvec build_areas(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    static arma::fvec areas;  // N  face areas
    if (areas.empty()) {
        areas.set_size(f.size());
        for (size_t i = 0; i < f.size(); i++) {
            const auto &facet = f[i];
            auto v1 = arma::fvec(const_cast<float *>(v[facet.vs[0]].data()), 3, false, true);
            auto v2 = arma::fvec(const_cast<float *>(v[facet.vs[1]].data()), 3, false, true);
            auto v3 = arma::fvec(const_cast<float *>(v[facet.vs[2]].data()), 3, false, true);

            float area = arma::norm(arma::cross(v2 - v1, v3 - v1));
            areas[i] = area;
        }
    }
    return areas;
}

TriangleMesh::TriangleMesh() {
}

TriangleMesh::~TriangleMesh() {
}

TriangleMesh::TriangleMesh(const TriangleMesh &t)
    : verts(t.verts), face_idxs(t.face_idxs) {

}

TriangleMesh::TriangleMesh(TriangleMesh &&t)
    : verts(std::move(t.verts)), face_idxs(std::move(t.face_idxs)) {

}

TriangleMesh::TriangleMesh(std::vector<xyz>&& v, std::vector<face_idx_t>&& f)
    : verts(std::move(v)), face_idxs(std::move(f)) {

}

std::vector<float> TriangleMesh::geodesic(const std::vector<int> &source) {
    auto adjs = build_adjs(this->verts, this->face_idxs);
    auto areas = build_areas(this->verts, this->face_idxs);
}