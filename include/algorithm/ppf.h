#ifndef PPF_H
#define PPF_H

#include "geometry/pointcloud.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
class PPF
{
private:
    thrust::device_vector<uint64_t> ppf_codes;
public:
    PPF();
    ~PPF();
    
    void setup_model(const PointCloud &model);
};




#endif