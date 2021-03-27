#ifndef PPF_H
#define PPF_H

#include "geometry/pointcloud.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <math.h>
#include <Eigen/Dense>


struct Pose
{
    uint32_t vote = 0;
    float r[9];
    float t[3];
    __host__ __device__ Pose() {}
    __host__ __device__ Pose(uint32_t vote, const float *r, const float *t) {
        this->vote = vote;
        if (r) memcpy(this->r, r, sizeof(float) * 9);
        if (t) memcpy(this->t, t, sizeof(float) * 3);
    }
    __host__ __device__ Pose(const Pose &t) {
        this->vote = t.vote;
        memcpy(this->r, t.r, sizeof(float) * 9);
        memcpy(this->t, t.t, sizeof(float) * 3);
    }
};



class PPF
{
private:
    thrust::device_vector<uint64_t> model_ppf_codes;
    thrust::device_vector<float3> model_pc, model_pc_normal;
    thrust::device_vector<float> model_transforms;
    thrust::device_vector<uint32_t> key2ppf, model_hash_keys, ppf_count, first_ppf_idx;
    float dist_delta, angle_delta;
    float cluster_dist_th, cluster_angle_th;
    int min_vote_th;
public:
    PPF(float dist_delta, float angle_delta, 
        float cluster_dist_th = 0.05f, float cluster_angle_th = 24.f / 180.f * M_PI, 
        int min_vote_th = 5);
    ~PPF();
    
    void setup_model(std::shared_ptr<PointCloud> model);
    std::vector<Pose> detect(std::shared_ptr<PointCloud> scene);
};




#endif