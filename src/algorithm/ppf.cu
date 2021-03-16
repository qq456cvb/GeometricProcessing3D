#include "algorithm/ppf.h"
#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>


PPF::PPF()
{
}

PPF::~PPF()
{
}

struct PPFKernel
{
    float3 *pc;
    uint64_t *ppf_codes;
    int npoints;

    PPFKernel(const thrust::device_vector<float> &pc, thrust::device_vector<uint64_t> &ppf_codes, const int npoints) : 
        pc((float3 *)thrust::raw_pointer_cast(&pc[0])), 
        ppf_codes(thrust::raw_pointer_cast(&ppf_codes[0])),
        npoints(npoints) {}

    __device__ void operator()(const uint64_t &t) const {
        int a = t / npoints;
        int b = t % npoints;
        if (pc[a].x > pc[b].x) ppf_codes[t] = 1;
        else ppf_codes[t] = 0;
    }
};


void PPF::setup_model(const PointCloud &model) {
    thrust::device_vector<float> pc(reinterpret_cast<const float*>(&(*model.verts.begin())), reinterpret_cast<const float*>(&(*model.verts.end())));
    int npoints = static_cast<int>(pc.size() / 3);
    thrust::device_vector<uint64_t> ppf_codes{model.verts.size() * model.verts.size(), 0};

    PPFKernel kern(pc, ppf_codes, npoints);
    thrust::for_each_n(thrust::counting_iterator<size_t>(0), ppf_codes.size(), kern);
    auto sum = thrust::reduce(ppf_codes.begin(), ppf_codes.end());
    printf("%lu, %lu\n", ppf_codes.size(), sum);
}