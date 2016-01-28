#include "cuda_kernels_common.cuh"

template <typename T>
void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, T* ptr, T value, int num_samples) {
    MemsetKernel<cuComplex><<<grid_size, block_size, 0, stream>>>(ptr, value, num_samples);
}

void launch_MultiplyFftKernel(int grid_size, int block_size, cudaStream_t stream, cufftComplex* time_proj_fft, const cufftComplex* filter_fft, int num_samples) {
    MultiplyFftKernel<<<grid_size, block_size, 0, stream>>>(time_proj_fft, filter_fft, num_samples);
}

// explicit function template instantiations for required datatypes
template void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* ptr, cuComplex value, int num_samples);
