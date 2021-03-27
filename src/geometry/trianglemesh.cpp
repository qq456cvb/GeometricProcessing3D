#include "geometry/trianglemesh.h"
#include <math.h>
#include <Eigen/Sparse>


static inline float calc_cot(const Eigen::Vector3f &a, const Eigen::Vector3f  &b) {
    return a.dot(b) / a.cross(b).norm();
}

auto build_v2v_nbrs(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    std::vector<std::vector<int>> adjs;
    adjs.resize(v.size());
    for (auto &facet : f) {
        const auto &vs = facet.vs;
        adjs[vs[0]].push_back(vs[1]);
        adjs[vs[0]].push_back(vs[2]);
        adjs[vs[1]].push_back(vs[0]);
        adjs[vs[1]].push_back(vs[2]);
        adjs[vs[2]].push_back(vs[0]);
        adjs[vs[2]].push_back(vs[1]);
    }
    return adjs;
}

auto build_areas(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    std::vector<float> areas;  // N  face areas
    areas.resize(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        const auto &facet = f[i];
        const auto &v1 = v[facet.vs[0]];
        const auto &v2 = v[facet.vs[1]];
        const auto &v3 = v[facet.vs[2]];

        float area = (v2 - v1).cross(v3 - v1).norm();
        areas[i] = area;
    }
    return areas;
}

auto build_face_ns(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    std::vector<xyz> face_ns;
    face_ns.resize(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        const auto &facet = f[i];
        const auto &v1 = v[facet.vs[0]];
        const auto &v2 = v[facet.vs[1]];
        const auto &v3 = v[facet.vs[2]];

        Eigen::Vector3f n = (v2 - v1).cross(v3 - v1).normalized();
        face_ns[i] = { n(0), n(1), n(2) };
    }
    return face_ns;
}


auto build_v2f_nbrs(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    std::vector<std::vector<int>> adjs;
    adjs.resize(v.size());
    for (size_t i = 0; i < f.size(); i++) {
        const auto &facet = f[i];
        adjs[facet.vs[0]].push_back(i);
        adjs[facet.vs[1]].push_back(i);
        adjs[facet.vs[2]].push_back(i);
    }
    return adjs;
}

auto build_e2f_nbrs(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    pair_map adjs{0, [](const std::pair<int, int> &p) {
        return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
    }, [] (const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
        return p1 == p2;
    }};
    for (size_t i = 0; i < f.size(); i++) {
        const auto &facet = f[i];
        const auto &vs = facet.vs;
        adjs[std::make_pair(vs[0], vs[1])].push_back(i);
        adjs[std::make_pair(vs[0], vs[2])].push_back(i);
        adjs[std::make_pair(vs[1], vs[2])].push_back(i);
        adjs[std::make_pair(vs[1], vs[0])].push_back(i);
        adjs[std::make_pair(vs[2], vs[0])].push_back(i);
        adjs[std::make_pair(vs[2], vs[1])].push_back(i);
    }
    return adjs;
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
    if (v2v_nbrs.empty()) v2v_nbrs = build_v2v_nbrs(this->verts, this->face_idxs);
    if (areas.empty()) areas = build_areas(this->verts, this->face_idxs);
    if (v2f_nbrs.empty()) v2f_nbrs = build_v2f_nbrs(this->verts, this->face_idxs);
    if (e2f_nbrs.empty()) e2f_nbrs = build_e2f_nbrs(this->verts, this->face_idxs);
    if (face_ns.empty()) face_ns = build_face_ns(this->verts, this->face_idxs);

    Eigen::VectorXf u0 = Eigen::VectorXf::Zero(verts.size());
    u0.resize(verts.size());
    for (const auto &s : source) {
        u0(s) = 1.f;
    }

    Eigen::SparseMatrix<float> L(verts.size(), verts.size());
    for (const auto &nbrs : e2f_nbrs) {
        const auto &edge = nbrs.first;
        int i = edge.first, j = edge.second;
        const auto &faces = nbrs.second;
        for (size_t c = 0; c < faces.size(); c++) {
            for (auto v : face_idxs[faces[c]].vs) {
                if (v != i && v != j) {
                    auto vi = verts[i] - verts[v];
                    auto vj = verts[j] - verts[v];
                    float cot = calc_cot(vi, vj);
                    L.coeffRef(i, j) += 0.5f * cot;
                    L.coeffRef(j, i) += 0.5f * cot;
                    break;
                }
            }
        }
    }
    Eigen::VectorXf rowsum = L * Eigen::VectorXf::Ones(L.cols());
    for (size_t i = 0; i < L.cols(); i++) {
        L.coeffRef(i, i) -= rowsum(i);
    }

    Eigen::SparseMatrix<float> A(verts.size(), verts.size());
    for (size_t i = 0; i < verts.size(); i++) {
        float a = 0.f;
        for (int f_idx : v2f_nbrs[i]) {
            a += areas[f_idx];
        }
        A.insert(i, i) = a / 3.f;
    }

    // determine spacing h^2
    float t = 0;
    for (const auto &nbrs : e2f_nbrs) {
        const auto &edge = nbrs.first;
        int i = edge.first, j = edge.second;
        auto ij = verts[j] - verts[i];
        t += ij.norm();
    }
    t /= e2f_nbrs.size();
    t *= t;

    Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A - t * L);
    Eigen::VectorXf u = solver.solve(u0);
    Eigen::MatrixXf grad_u = Eigen::MatrixXf::Zero(face_idxs.size(), 3);
    for (size_t i = 0; i < face_idxs.size(); i++) {
        const auto &vs = face_idxs[i].vs;
        Eigen::Vector3f grad(0.f, 0.f, 0.f);
        for (size_t j = 0; j < 3; j++) {
            int v1_idx = vs[j];
            int v2_idx = vs[(j + 1) % 3];
            int v3_idx = vs[(j + 2) % 3];
            Eigen::Vector3f edge = verts[v2_idx] - verts[v1_idx];
            grad += u(v3_idx) * face_ns[i].cross(edge);
        }
        grad_u.row(i) = -(grad / (2.f * areas[i])).normalized().transpose();
    }

    // maybe speedup by precompute angles
    Eigen::VectorXf div = Eigen::VectorXf::Zero(verts.size());
    for (size_t i = 0; i < verts.size(); i++) {
        float sum = 0;
        for (auto f_idx : v2f_nbrs[i]) {
            const auto &vs = face_idxs[f_idx].vs;
            int ref = -1;
            for (size_t j = 0; j < 3; j++) {
                if (vs[j] == i) {
                    ref = static_cast<int>(j);
                    break;
                }
            }
            int v1_idx = vs[ref];
            int v2_idx = vs[(ref + 1) % 3];
            int v3_idx = vs[(ref + 2) % 3];
            auto e1 = verts[v2_idx] - verts[v1_idx];
            auto e2 = verts[v3_idx] - verts[v2_idx];
            auto e3 = verts[v1_idx] - verts[v3_idx];

            sum += 0.5f * (calc_cot(-e2, e3) * e1.dot(grad_u.row(f_idx).transpose()) + calc_cot(-e1, e2) * (-e3).dot(grad_u.row(f_idx).transpose()));
        }
        div(i) = sum;
    }
    solver.compute(L);
    Eigen::VectorXf phi = solver.solve(div);
    auto phi_min = phi.minCoeff();
    std::for_each(phi.data(), phi.data() + phi.size(), [phi_min](auto& val) { val -= phi_min; } ); 
    std::vector<float> res{phi.data(), phi.data() + phi.size()};
    
    return res;
}