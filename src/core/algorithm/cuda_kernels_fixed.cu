#include "cuda_kernels_internal.cuh"

// Slice a 3D lookup table through plane defined by two unit vectors.
// X- and y-components of grid determines the number of samples.
// NOTE: Number of threads in each block should be one.
__global__ void SliceLookupTable(float3 origin,
                                 float3 dir0,
                                 float3 dir1,
                                 float* output,
                                 cudaTextureObject_t lut_tex) {
    const int global_idx = blockIdx.x*gridDim.x + blockIdx.y;
    
    // FORMULA: offset = dim0*num_samples1 + dim1
    const int idx0 = blockIdx.x;  // idx0 = 0..gridDim.x
    const int idx1 = blockIdx.y;  // idx1 = 1..gridDim.y

    // Map to normalized distance in [0.0, 1.0]
    const auto normalized_dist0 = static_cast<float>(idx0)/(gridDim.x-1);
    const auto normalized_dist1 = static_cast<float>(idx1)/(gridDim.y-1);

    const auto tex_pos = origin + dir0*normalized_dist0 + dir1*normalized_dist1;
    output[global_idx] = tex3D<float>(lut_tex, tex_pos.x, tex_pos.y, tex_pos.z);
}
