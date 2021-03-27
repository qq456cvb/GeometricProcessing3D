#include "algorithm/ppf.h"
#include <cuda_runtime.h>
#include "utils/helper_math.h"
#include <Eigen/Dense>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/count.h>
#include <thrust/sequence.h>

#define GPU_REDUCE

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


PPF::PPF(float dist_delta, float angle_delta, 
        float cluster_dist_th, float cluster_angle_th,
        int min_vote_th) :
    dist_delta(dist_delta),
    angle_delta(angle_delta),
    cluster_dist_th(cluster_dist_th),
    cluster_angle_th(cluster_angle_th),
    min_vote_th(min_vote_th)
{
}

PPF::~PPF()
{
}


std::vector<float3> convert2continuous(const std::vector<xyz> &x) {
    std::vector<float3> res(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        memcpy(&res[i], x[i].data(), sizeof(float) * 3);
    }
    return res;
}


void PPF::setup_model(std::shared_ptr<PointCloud> model) {
    std::vector<float3> cv = convert2continuous(model->verts);
    std::vector<float3> cn = convert2continuous(model->normals);
    model_pc = thrust::device_vector<float3>(cv.begin(), cv.end());
    model_pc_normal = thrust::device_vector<float3>(cn.begin(), cn.end());
    // float3 *pc_ptr = thrust::raw_pointer_cast(model_pc.data());
    // float3 *pc_normal_ptr = thrust::raw_pointer_cast(model_pc_normal.data());
    std::array<float, 3> center = {0, 0, 0};
    for (const auto &v : model->verts) {
        center[0] += v[0];
        center[1] += v[1];
        center[2] += v[2];
    } 
    
    int npoints = static_cast<int>(model_pc.size());

    model_transforms.resize(npoints * 9);
    // float *transforms_ptr = thrust::raw_pointer_cast(transforms.data());
    TransXKernel transx_kern(model_pc_normal, model_transforms);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_pc_normal.size(), transx_kern);

    model_ppf_codes.resize(model->verts.size() * model->verts.size());
    PPFKernel ppf_kern(model_pc, model_pc_normal, model_transforms, model_ppf_codes, npoints, dist_delta, angle_delta);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_ppf_codes.size(), ppf_kern);

    thrust::sort(model_ppf_codes.begin(), model_ppf_codes.end());

    key2ppf.resize(model_ppf_codes.size());
    thrust::device_vector<uint32_t> hash_keys{model_ppf_codes.size(), 0};
    uint32_t *key2ppf_ptr = thrust::raw_pointer_cast(key2ppf.data());
    uint32_t *hash_keys_ptr = thrust::raw_pointer_cast(hash_keys.data());
    uint64_t *ppf_codes_ptr = thrust::raw_pointer_cast(model_ppf_codes.data());
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), model_ppf_codes.size(), [=] __device__ (int i) {
        key2ppf_ptr[i] = static_cast<uint32_t>(0x3FFFFFF & ppf_codes_ptr[i] >> 6);
        hash_keys_ptr[i] = static_cast<uint32_t>(ppf_codes_ptr[i] >> 32);
    });

    reduce_by_key<uint32_t>(hash_keys, model_hash_keys, ppf_count);

    first_ppf_idx.resize(ppf_count.size());
    thrust::exclusive_scan(ppf_count.begin(), ppf_count.end(), first_ppf_idx.begin());
}


std::vector<Pose> PPF::detect(std::shared_ptr<PointCloud> scene) {
    auto start = std::chrono::high_resolution_clock::now();
    // TODO: filter points and pairs
    // thrust::device_vector<float3> pc(reinterpret_cast<const float3*>(&(*scene->verts.begin())), reinterpret_cast<const float3*>(&(*scene->verts.end())));
    // thrust::device_vector<float3> pc_normal(reinterpret_cast<const float3*>(&(*scene->normals.begin())), reinterpret_cast<const float3*>(&(*scene->normals.end())));
    std::vector<float3> cv = convert2continuous(scene->verts);
    std::vector<float3> cn = convert2continuous(scene->normals);
    thrust::device_vector<float3> pc = thrust::device_vector<float3>(cv.begin(), cv.end());
    thrust::device_vector<float3> pc_normal = thrust::device_vector<float3>(cn.begin(), cn.end());

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
        uint32_t scene_idx = static_cast<uint32_t>(0x3FFFFFF & ppf_codes_ptr[i] >> 6);
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
    printf("unique votes shape: %lu\n", unique_votes.size());

    auto stop = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time1: " << duration.count() << std::endl; 

#ifdef GPU_REDUCE
    thrust::device_vector<uint32_t> scene_idxs(unique_votes.size(), 0);
    thrust::device_vector<uint32_t> model_idxs(unique_votes.size(), 0);
    thrust::device_vector<Pose> origin_poses(unique_votes.size());
    uint32_t *vote_counts_ptr = thrust::raw_pointer_cast(vote_counts.data());
    float *model_transforms_ptr = thrust::raw_pointer_cast(model_transforms.data());
    float *scene_transforms_ptr = thrust::raw_pointer_cast(transforms.data());

    float3 *model_pc_ptr = thrust::raw_pointer_cast(model_pc.data());
    float3 *scene_pc_ptr = thrust::raw_pointer_cast(pc.data());
    uint64_t *unique_votes_ptr = thrust::raw_pointer_cast(unique_votes.data());
    Pose *origin_poses_ptr = thrust::raw_pointer_cast(origin_poses.data());
    uint32_t *scene_idxs_ptr = thrust::raw_pointer_cast(scene_idxs.data());
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), unique_votes.size(), [=] __device__ (const size_t &i) {
        const auto &vote = unique_votes_ptr[i];
        uint32_t scene_idx = static_cast<uint32_t>(vote >> 32);
        scene_idxs_ptr[i] = scene_idx;
        uint32_t model_idx = static_cast<uint32_t>(0x3FFFFFF & vote >> 6);

        Pose p;
        p.vote = vote_counts_ptr[i];
        float *model_trans = &model_transforms_ptr[model_idx * 9];
        float *scene_trans = &scene_transforms_ptr[scene_idx * 9];

        float3 p1 = model_pc_ptr[model_idx];
        float3 p2 = scene_pc_ptr[scene_idx];
            
#define a scene_trans
#define b model_trans
        p.r[0] = a[0] * b[0] + a[3] * b[3] + a[6] * b[6];
        p.r[1] = a[0] * b[1] + a[3] * b[4] + a[6] * b[7];
        p.r[2] = a[0] * b[2] + a[3] * b[5] + a[6] * b[8];
        p.r[3] = a[1] * b[0] + a[4] * b[3] + a[7] * b[6];
        p.r[4] = a[1] * b[1] + a[4] * b[4] + a[7] * b[7];
        p.r[5] = a[1] * b[2] + a[4] * b[5] + a[7] * b[8];
        p.r[6] = a[2] * b[0] + a[5] * b[3] + a[8] * b[6];
        p.r[7] = a[2] * b[1] + a[5] * b[4] + a[8] * b[7];
        p.r[8] = a[2] * b[2] + a[5] * b[5] + a[8] * b[8];
#undef a
#undef b   
        float3 rp1 = make_float3(dot(make_float3(p.r[0], p.r[1], p.r[2]), p1), dot(make_float3(p.r[3], p.r[4], p.r[5]), p1), dot(make_float3(p.r[6], p.r[7], p.r[8]), p1));
        float3 t = p2 - rp1;
        p.t[0] = t.x;
        p.t[1] = t.y;
        p.t[2] = t.z;
        origin_poses_ptr[i] = p;
    });

    thrust::device_vector<Pose> unique_poses(unique_votes.size());
    auto unique_value_begin = thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<size_t>(0), unique_poses.begin()));
    auto end = thrust::reduce_by_key(scene_idxs.begin(), scene_idxs.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<size_t>(0), origin_poses.begin())), 
        thrust::make_discard_iterator(), 
        unique_value_begin, 
        thrust::equal_to<uint32_t>(),
        [=] __device__ (const thrust::tuple<size_t, Pose> &t1, const thrust::tuple<size_t, Pose> &t2) {
            const auto &pose_1 = thrust::get<1>(t1);
            const auto &pose_2 = thrust::get<1>(t2);
            
            if (pose_1.vote > pose_2.vote) {
                return thrust::make_tuple(thrust::get<0>(t1), Pose(pose_1));
            } else {
                return thrust::make_tuple(thrust::get<0>(t2), Pose(pose_2));
            }
        });
    
    unique_poses.resize(thrust::distance(unique_value_begin, end.second));

    thrust::sort(unique_poses.begin(), unique_poses.end(), [] __device__ (const Pose &p1, const Pose &p2) {
        return p1.vote > p2.vote;
    });
    std::vector<Pose> poses;
    poses.resize(unique_poses.size());
    thrust::copy(unique_poses.begin(), unique_poses.end(), poses.begin());
    
#else
    // TODO test if gpu reduce is faster, current about 10 ms
    start = stop;
    thrust::host_vector<float> h_model_transforms(model_transforms);
    thrust::host_vector<float> h_scene_transforms(transforms);
    thrust::host_vector<uint32_t> h_vote_counts(vote_counts);
    // printf("sanity check: %u\n", thrust::reduce(vote_counts.begin(), vote_counts.end(), 0));
    thrust::host_vector<uint64_t> h_votes(unique_votes);
    thrust::host_vector<float3> h_model_pc(model_pc);
    thrust::host_vector<float3> h_scene_pc(pc);
    uint32_t curr_scene_idx = static_cast<uint32_t>(h_votes[0] >> 32), curr_vote = 0;
    arma::fmat33 curr_r_t;
    arma::fvec3 curr_t;
    std::vector<Pose> poses;
    for (size_t i = 0; i < h_votes.size(); i++)
    {
        uint32_t scene_idx = static_cast<uint32_t>(h_votes[i] >> 32);
        uint32_t vote = h_vote_counts[i];
        if (scene_idx != curr_scene_idx) {
            if (curr_vote > min_vote_th) {
                poses.emplace_back(curr_vote, curr_r_t.memptr(), curr_t.memptr());
            }
            curr_vote = 0;
            curr_scene_idx = scene_idx;
        } else {
            if (vote > curr_vote) {
                uint64_t model_idx = static_cast<uint64_t>(0x3FFFFFF & h_votes[i] >> 6);
                curr_vote = vote;
                arma::fmat model_trans_t(const_cast<float *>(&h_model_transforms[model_idx * 9]), 3, 3, false, true);  // col major // potential overflow
                arma::fmat scene_trans_t(const_cast<float *>(&h_scene_transforms[scene_idx * 9]), 3, 3, false, true);
                curr_r_t = model_trans_t * scene_trans_t.t();
                // strange: buggy below
                arma::fvec p2((float *)(&h_scene_pc[scene_idx]), 3, false, true);
                arma::fvec p1((float *)(&h_model_pc[model_idx]), 3, false, true);
                curr_t = p2 - curr_r_t.t() * p1;  // strange armadillo bug: must use t() twice
            }
        }
    }

    std::sort(poses.begin(), poses.end(), [](const Pose &p1, const Pose &p2) {
        return p1.vote > p2.vote;
    });

#endif
    stop = std::chrono::high_resolution_clock::now(); 
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time2: " << duration.count() << std::endl; 

    printf("peak finding: %lu\n", poses.size());
    

    // clustering, has to be on cpu, serial
    std::vector<std::pair<int, int> > cluster_votes;
    std::vector<std::pair<int, std::vector<Pose>>> pose_clusters;
    int cnt = 0;
    for (const auto& pose : poses) {
        bool found_cluster = false;
        for (auto& cluster : pose_clusters) {
            for (auto &cpose : cluster.second) {
                float3 t_dist = make_float3(pose.t[0] - cpose.t[0], pose.t[1] - cpose.t[1], pose.t[2] - cpose.t[2]);
                float trace = pose.r[0] * cpose.r[0] + pose.r[3] * cpose.r[3] + pose.r[6] * cpose.r[6]
                    + pose.r[1] * cpose.r[1] + pose.r[4] * cpose.r[4] + pose.r[7] * cpose.r[7]
                    + pose.r[2] * cpose.r[2] + pose.r[5] * cpose.r[5] + pose.r[8] * cpose.r[8];
                if (dot(t_dist, t_dist) < cluster_dist_th * cluster_dist_th && acosf(trace * .5f - .5f) < cluster_angle_th) {
                    found_cluster = true;
                    cluster.second.push_back(pose);
                    cluster.first += pose.vote;
                    break;
                }
            }
            if (found_cluster) break;
        }
        cnt += 1;

        // Add a new cluster of poses
        if (!found_cluster) {
            std::vector<Pose> cluster = { pose };
            pose_clusters.emplace_back(pose.vote, cluster);
        }
    }

    std::sort(pose_clusters.begin(), pose_clusters.end(), [](const auto &c1, const auto &c2) {
        return c1.first > c2.first;
    });

    // TODO: merge cluster poses
    printf("final pose clusters: %lu\n", pose_clusters.size());
    for (size_t i = 0; i < std::min(pose_clusters.size(), size_t(20)); i++)
    {
        std::cout << i << ", " << pose_clusters[i].first << std::endl 
            << Eigen::Map<Eigen::Matrix3f>((float *)pose_clusters[i].second[0].r).transpose() << std::endl
            << Eigen::Map<Eigen::Vector3f>((float *)pose_clusters[i].second[0].t).transpose() << std::endl;
    }
    
    stop = std::chrono::high_resolution_clock::now(); 
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "time3: " << duration.count() << std::endl; 

    std::vector<Pose> results;
    for (const auto &cluster: pose_clusters) {
        results.emplace_back(cluster.first, cluster.second[0].r, cluster.second[0].t);
    }
    return results;
}