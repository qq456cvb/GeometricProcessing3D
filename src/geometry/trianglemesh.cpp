#include "geometry/trianglemesh.h"
#include <math.h>


#define ARMA_FVEC(x) arma::fvec(const_cast<float *>(x.data()), x.size(), false, true)
#define ARMA_FVEC3(x) arma::fvec(const_cast<float *>(x.data()), 3, false, true)

static inline float calc_cot(const arma::fvec &a, const arma::fvec &b) {
    return arma::dot(a, b) / arma::norm(arma::cross(a, b));
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
        auto v1 = ARMA_FVEC3(v[facet.vs[0]]);
        auto v2 = ARMA_FVEC3(v[facet.vs[1]]);
        auto v3 = ARMA_FVEC3(v[facet.vs[2]]);

        float area = arma::norm(arma::cross(v2 - v1, v3 - v1));
        areas[i] = area;
    }
    return areas;
}

auto build_face_ns(const std::vector<xyz> &v, const std::vector<face_idx_t> &f) {
    std::vector<xyz> face_ns;
    face_ns.resize(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        const auto &facet = f[i];
        auto v1 = ARMA_FVEC3(v[facet.vs[0]]);
        auto v2 = ARMA_FVEC3(v[facet.vs[1]]);
        auto v3 = ARMA_FVEC3(v[facet.vs[2]]);

        arma::fvec3 n = arma::normalise(arma::cross(v2 - v1, v3 - v1));
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

    arma::fvec u0(verts.size(), arma::fill::zeros);
    u0.resize(verts.size());
    for (const auto &s : source) {
        u0(s) = 1.f;
    }

    arma::sp_fmat L(verts.size(), verts.size());
    for (const auto &nbrs : e2f_nbrs) {
        const auto &edge = nbrs.first;
        int i = edge.first, j = edge.second;
        const auto &faces = nbrs.second;
        for (size_t c = 0; c < faces.size(); c++) {
            for (auto v : face_idxs[faces[c]].vs) {
                if (v != i && v != j) {
                    arma::fvec vi = ARMA_FVEC3(verts[i]) - ARMA_FVEC3(verts[v]);
                    arma::fvec vj = ARMA_FVEC3(verts[j]) - ARMA_FVEC3(verts[v]);
                    float cot = calc_cot(vi, vj);
                    L(i, j) += 0.5f * cot;
                    L(j, i) += 0.5f * cot;
                    break;
                }
            }
        }
    }
    L -= arma::diagmat(arma::sum(L, 1));

    arma::sp_fmat A(verts.size(), verts.size());
    for (size_t i = 0; i < verts.size(); i++) {
        float a = 0.f;
        for (int f_idx : v2f_nbrs[i]) {
            a += areas[f_idx];
        }
        A(i, i) = a / 3.f;
    }

    // determine spacing h^2
    float t = 0;
    for (const auto &nbrs : e2f_nbrs) {
        const auto &edge = nbrs.first;
        int i = edge.first, j = edge.second;
        arma::fvec ij = ARMA_FVEC3(verts[j]) - ARMA_FVEC3(verts[i]);
        t += arma::norm(ij);
    }
    t /= e2f_nbrs.size();
    t *= t;

    arma::fvec u = arma::spsolve(A - t * L, u0, "lapack");
    arma::fmat grad_u(face_idxs.size(), 3, arma::fill::zeros);
    for (size_t i = 0; i < face_idxs.size(); i++) {
        const auto &vs = face_idxs[i].vs;
        arma::fvec3 grad;
        grad.fill(0);
        for (size_t j = 0; j < 3; j++) {
            int v1_idx = vs[j];
            int v2_idx = vs[(j + 1) % 3];
            int v3_idx = vs[(j + 2) % 3];
            arma::fvec edge = ARMA_FVEC3(verts[v2_idx]) - ARMA_FVEC3(verts[v1_idx]);
            grad += u(v3_idx) * arma::cross(ARMA_FVEC3(face_ns[i]), edge);
        }
        grad_u.row(i) = -arma::normalise(grad / (2.f * areas[i])).t();
    }

    // maybe speedup by precompute angles
    arma::fvec div(verts.size(), arma::fill::zeros);
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
            arma::fvec e1 = ARMA_FVEC3(verts[v2_idx]) - ARMA_FVEC3(verts[v1_idx]);
            arma::fvec e2 = ARMA_FVEC3(verts[v3_idx]) - ARMA_FVEC3(verts[v2_idx]);
            arma::fvec e3 = ARMA_FVEC3(verts[v1_idx]) - ARMA_FVEC3(verts[v3_idx]);

            sum += 0.5f * (calc_cot(-e2, e3) * arma::dot(e1, grad_u.row(f_idx).t()) + calc_cot(-e1, e2) * arma::dot(-e3, grad_u.row(f_idx).t()));
        }
        div(i) = sum;
    }
    arma::fvec phi = arma::spsolve(L, div, "lapack");
    auto phi_min = arma::min(phi);
    phi.for_each( [phi_min](auto& val) { val -= phi_min; } ); 
    std::vector<float> res{phi.begin(), phi.end()};
    
    return res;
}