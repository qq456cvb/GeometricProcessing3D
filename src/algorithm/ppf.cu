#include "algorithm/ppf.h"
#include <cuda_runtime.h>
#include "utils/helper_math.h"
#include <armadillo>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>


inline __device__ uint32_t rotl32(uint32_t x, int8_t r) {
  return (x << r) | (x >> (32 - r));
}


inline __device__ float angle_between(const float3 &a, const float3 &b) {
    return atan2(length(cross(a, b)), dot(a, b));
}


inline __device__ uint32_t murmurppf(const uint32_t ppf[4]) {
    uint32_t h1 = 42;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    for (int i = 0; i < 4; i++) {
        uint32_t k1 = ppf[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13); 
        h1 = h1 * 5 + 0xe6546b64;
    }

    h1 ^= 16;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}


__device__ uint32_t compute_ppfhash (
    const float3& p1, const float3& n1,
    const float3& p2, const float3& n2,
    const float dist_delta, const float angle_delta) {

    // Compute the vector between the points
    float3 d = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

    // Compute the 4 components of the ppf feature
    const float f1 = length(d);
    const float f2 = angle_between(d, n1);
    const float f3 = angle_between(d, n2);
    const float f4 = angle_between(n1, n2);

    // Discretize the PPF Feature before hashing
    uint32_t feature[4];
    feature[0] = static_cast<uint32_t>(f1 / dist_delta);
    feature[1] = static_cast<uint32_t>(f2 / angle_delta);
    feature[2] = static_cast<uint32_t>(f3 / angle_delta);
    feature[3] = static_cast<uint32_t>(f4 / angle_delta);

    // Return the hash of the feature.
    return murmurppf(feature);
}


struct PPFKernel
{
    const float3 *pc, *pc_normal, *transforms;
    uint64_t *ppf_codes;
    int npoints;
    float dist_delta, angle_delta;

    PPFKernel(const thrust::device_vector<float3> &pc, 
            const thrust::device_vector<float3> &pc_normal, 
            const thrust::device_vector<float> &transforms, 
            thrust::device_vector<uint64_t> &ppf_codes, 
            const int npoints,
            const float dist_delta,
            const float angle_delta) : 
        pc(thrust::raw_pointer_cast(&pc[0])), 
        pc_normal(thrust::raw_pointer_cast(&pc_normal[0])),
        transforms((float3 *)thrust::raw_pointer_cast(&transforms[0])),
        ppf_codes(thrust::raw_pointer_cast(&ppf_codes[0])),
        npoints(npoints),
        dist_delta(dist_delta),
        angle_delta(angle_delta) {}

    __device__ void operator()(const uint64_t &n) const {
        int a = n / npoints;
        int b = n % npoints;
        const float3 *t = &transforms[a * 3];  // transform point a to axis x
        uint32_t ppf_hash = compute_ppfhash(pc[a], pc_normal[a], pc[b], pc_normal[b], dist_delta, angle_delta);
        float angle = atan2(-dot(t[2], pc[b]), dot(t[1], pc[b])); // in PPF paper, left-handed coordinates
        uint64_t angle_bin = static_cast<uint64_t>(angle / angle_delta);
        uint64_t code = (static_cast<uint64_t>(ppf_hash) << 32) | 
                (static_cast<uint64_t>(a) << 6) | 
                angle_bin;

        // Save the code
        ppf_codes[n] = code;
    }
};


struct TransXKernel
{
    const float3 *pc_normal;
    float *transforms;
    TransXKernel(const thrust::device_vector<float3> &pc_normal,
            thrust::device_vector<float> &transforms) :
        pc_normal(thrust::raw_pointer_cast(&pc_normal[0])),
        transforms(thrust::raw_pointer_cast(&transforms[0])) {}

    __device__ void operator()(int i) const {
        const auto &n = pc_normal[i];
        auto y2 = n.y * n.y;
        auto z2 = n.z * n.z;
        auto yz = n.y * n.z;
        auto y2z2 = y2 + z2;

        // TODO: what if y2z2 == 0
        transforms[i * 9] = n.x;
        transforms[i * 9 + 1] = n.y;
        transforms[i * 9 + 2] = n.z;
        transforms[i * 9 + 3] = -n.y;
        transforms[i * 9 + 4] = 1 + (n.x - 1) * y2 / y2z2;
        transforms[i * 9 + 5] = (n.x - 1) * yz / y2z2;
        transforms[i * 9 + 6] = -n.z;
        transforms[i * 9 + 7] = transforms[i * 9 + 5];
        transforms[i * 9 + 8] = 1 + (n.x - 1) * z2 / y2z2;
    }
};

template <typename T>
void reduce_by_key(const thrust::device_vector<T> &keys, thrust::device_vector<T> &unique_keys, thrust::device_vector<uint32_t> &counts) {
    thrust::equal_to<T> binary_pred;
    thrust::plus<uint32_t> binary_op;
    counts.resize(keys.size());
    unique_keys.resize(keys.size());
    auto end = thrust::reduce_by_key(keys.begin(), keys.end(), thrust::make_constant_iterator(1), unique_keys.begin(), counts.begin(), binary_pred, binary_op);
    unique_keys.resize(thrust::distance(unique_keys.begin(), end.first));
    counts.resize(thrust::distance(counts.begin(), end.second));
}


PPF::PPF(const float &dist_delta, const float &angle_delta) :
    dist_delta(dist_delta),
    angle_delta(angle_delta)
{
}

PPF::~PPF()
{
}


void PPF::setup_model(const PointCloud &model) {
    model_pc = thrust::device_vector<float3>(reinterpret_cast<const float3*>(&(*model.verts.begin())), reinterpret_cast<const float3*>(&(*model.verts.end())));
    model_pc_normal = thrust::device_vector<float3>(reinterpret_cast<const float3*>(&(*model.normals.begin())), reinterpret_cast<const float3*>(&(*model.normals.end())));
    // float3 *pc_ptr = thrust::raw_pointer_cast(model_pc.data());
    // float3 *pc_normal_ptr = thrust::raw_pointer_cast(model_pc_normal.data());
    int npoints = static_cast<int>(model_pc.size());

    model_transforms.resize(npoints * 9);
    // float *transforms_ptr = thrust::raw_pointer_cast(transforms.data());
    TransXKernel transx_kern(model_pc_normal, model_transforms);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_pc_normal.size(), transx_kern);
    // thrust::for_each_n(thrust::counting_iterator<size_t>(0), pc_normal.size(), [=] __device__ (int i) {
    //     const auto &n = pc_normal_ptr[i];
    //     auto y2 = n.y * n.y;
    //     auto z2 = n.z * n.z;
    //     auto yz = n.y * n.z;
    //     auto y2z2 = y2 + z2;

    //     // TODO: what if y2z2 == 0
    //     transforms_ptr[i * 9] = n.x;
    //     transforms_ptr[i * 9 + 1] = n.y;
    //     transforms_ptr[i * 9 + 2] = n.z;
    //     transforms_ptr[i * 9 + 3] = -n.y;
    //     transforms_ptr[i * 9 + 4] = 1 + (n.x - 1) * y2 / y2z2;
    //     transforms_ptr[i * 9 + 5] = (n.x - 1) * yz / y2z2;
    //     transforms_ptr[i * 9 + 6] = -n.z;
    //     transforms_ptr[i * 9 + 7] = transforms_ptr[i * 9 + 5];
    //     transforms_ptr[i * 9 + 8] = 1 + (n.x - 1) * z2 / y2z2;
    // });

    // thrust::host_vector<float> h_transforms(transforms);
    // for (size_t i = 0; i < 10; i++) {
    //     auto v = arma::fvec(const_cast<float *>(model.normals[i].data()), 3, false, true);
    //     auto R = arma::fmat(const_cast<float *>(&h_transforms[i * 9]), 3, 3, false, true);
    //     std::cout << R.t() * v << std::endl;
    // }

    model_ppf_codes.resize(model.verts.size() * model.verts.size());
    PPFKernel ppf_kern(model_pc, model_pc_normal, model_transforms, model_ppf_codes, npoints, dist_delta, angle_delta);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_ppf_codes.size(), ppf_kern);

    thrust::sort(model_ppf_codes.begin(), model_ppf_codes.end());

    key2ppf.resize(model_ppf_codes.size());
    thrust::device_vector<uint32_t> hash_keys{model_ppf_codes.size(), 0};
    uint32_t *key2ppf_ptr = thrust::raw_pointer_cast(key2ppf.data());
    uint32_t *hash_keys_ptr = thrust::raw_pointer_cast(hash_keys.data());
    uint64_t *ppf_codes_ptr = thrust::raw_pointer_cast(model_ppf_codes.data());
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_ppf_codes.size(), [=] __device__ (int i) {
        key2ppf_ptr[i] = static_cast<uint32_t>(0xffffffc0 & ppf_codes_ptr[i]);
        hash_keys_ptr[i] = static_cast<uint32_t>(ppf_codes_ptr[i] >> 32);
    });

    reduce_by_key<uint32_t>(hash_keys, model_hash_keys, ppf_count);

    first_ppf_idx.resize(ppf_count.size());
    thrust::exclusive_scan(ppf_count.begin(), ppf_count.end(), first_ppf_idx.begin());
}


void PPF::detect(const PointCloud &scene) {
    thrust::device_vector<float3> pc(reinterpret_cast<const float3*>(&(*scene.verts.begin())), reinterpret_cast<const float3*>(&(*scene.verts.end())));
    thrust::device_vector<float3> pc_normal(reinterpret_cast<const float3*>(&(*scene.normals.begin())), reinterpret_cast<const float3*>(&(*scene.normals.end())));

    int npoints = static_cast<int>(pc.size());
    int n_angle_bins = static_cast<int>(2.0000001 * M_PI / angle_delta);

    thrust::device_vector<float> transforms{npoints * 9, 0};
    TransXKernel transx_kern(pc_normal, transforms);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), pc_normal.size(), transx_kern);

    thrust::device_vector<uint64_t> ppf_codes{npoints * npoints, 0};
    PPFKernel ppf_kern(pc, pc_normal, transforms, ppf_codes, npoints, dist_delta, angle_delta);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), ppf_codes.size(), ppf_kern);

    thrust::device_vector<uint32_t> hash_keys{ppf_codes.size(), 0};
    uint32_t *hash_keys_ptr = thrust::raw_pointer_cast(hash_keys.data());
    uint64_t *ppf_codes_ptr = thrust::raw_pointer_cast(ppf_codes.data());
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), ppf_codes.size(), [=] __device__ (int i) {
        hash_keys_ptr[i] = static_cast<uint32_t>(ppf_codes_ptr[i] >> 32);
    });

    thrust::device_vector<uint32_t> nn_idx{hash_keys.size(), 0};  // uint32 big enough?
    thrust::lower_bound(model_hash_keys.begin(), model_hash_keys.end(), hash_keys.begin(), hash_keys.end(), nn_idx.begin());

    uint32_t *first_ppf_idx_ptr = thrust::raw_pointer_cast(first_ppf_idx.data());
    uint32_t *nn_idx_ptr = thrust::raw_pointer_cast(nn_idx.data());
    uint32_t *key2ppf_ptr = thrust::raw_pointer_cast(key2ppf.data());
    uint64_t *model_ppf_codes_ptr = thrust::raw_pointer_cast(model_ppf_codes.data());
    thrust::device_vector<uint64_t> votes{nn_idx.size(), 0};
    uint64_t *votes_ptr = thrust::raw_pointer_cast(votes.data());

    thrust::for_each_n(thrust::counting_iterator<size_t>(0), nn_idx.size(), [=] __device__ (int i) {
        uint32_t model_ppf_idx = first_ppf_idx_ptr[nn_idx_ptr[i]];
        uint32_t model_idx = key2ppf_ptr[model_ppf_idx];
        uint32_t scene_idx = static_cast<uint32_t>(0xffffffc0 & ppf_codes_ptr[i]);
        int scene_angle_bin = static_cast<int>(63 & ppf_codes_ptr[i]);
        int model_angle_bin = static_cast<int>(63 & model_ppf_codes_ptr[model_ppf_idx]);
        int angle_bin = scene_angle_bin - model_angle_bin;
        if (angle_bin < 0) angle_bin += n_angle_bins;
        votes_ptr[i] = (static_cast<uint64_t>(scene_idx) << 32) |
            (static_cast<uint64_t>(model_idx) << 6) |
            static_cast<uint64_t>(angle_bin);
    });

    thrust::sort(votes.begin(), votes.end());

    thrust::device_vector<uint64_t> unique_votes;
    thrust::device_vector<uint32_t> vote_counts;
    reduce_by_key<uint64_t>(votes, unique_votes, vote_counts);

    printf("max votes: %u\n", thrust::reduce(vote_counts.begin(), vote_counts.end(), 0, thrust::maximum<uint32_t>()));
}