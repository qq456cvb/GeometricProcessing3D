#ifndef PPF_H
#define PPF_H

#include "geometry/pointcloud.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <math.h>
#include <armadillo>
class PPF
{
private:
    thrust::device_vector<uint64_t> model_ppf_codes;
    thrust::device_vector<float3> model_pc, model_pc_normal;
    thrust::device_vector<float> model_transforms;
    thrust::device_vector<uint32_t> key2ppf, model_hash_keys, ppf_count, first_ppf_idx;
    float dist_delta, angle_delta;
    float cluster_dist_th = 0.3f, cluster_angle_th = 15.f / 180.f * M_PI;
    int min_vote_th = 5;
    arma::fvec3 model_center;
public:
    PPF(const float &dist_delta, const float &angle_delta);
    ~PPF();
    
    void setup_model(const PointCloud &model);
    void detect(const PointCloud &scene);
};




#endif