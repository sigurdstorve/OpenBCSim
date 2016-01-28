#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include "cuda_kernels_common.cuh"

template <typename T>
void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, T* ptr, T value, int num_samples) {
    MemsetKernel<cuComplex><<<grid_size, block_size, 0, stream>>>(ptr, value, num_samples);
}

template void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* ptr, cuComplex value, int num_samples);
