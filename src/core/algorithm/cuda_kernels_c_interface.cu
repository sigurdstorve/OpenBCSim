#include "cuda_kernels_c_interface.h"
#include "cuda_kernels_common.cuh"      // for common kernels
#include "cuda_kernels_fixed.cuh"       // for FixedAlgKernel
#include "cuda_kernels_spline1.cuh"     // for fixedAlg_updateConstantMemory_internal

template <typename T>
void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, T* ptr, T value, int num_samples) {
    MemsetKernel<cuComplex><<<grid_size, block_size, 0, stream>>>(ptr, value, num_samples);
}

void launch_MultiplyFftKernel(int grid_size, int block_size, cudaStream_t stream, cufftComplex* time_proj_fft, const cufftComplex* filter_fft, int num_samples) {
    MultiplyFftKernel<<<grid_size, block_size, 0, stream>>>(time_proj_fft, filter_fft, num_samples);
}

void launch_DemodulateKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* signal, float w, int num_samples) {
    DemodulateKernel<<<grid_size, block_size, 0, stream>>>(signal, w, num_samples);
}

void launch_ScaleSignalKernel(int grid_size, int block_size, cudaStream_t stream, cufftComplex* signal, float factor, int num_samples) {
    ScaleSignalKernel<<<grid_size, block_size, 0, stream>>>(signal, factor, num_samples);
}

template <bool A, bool B, bool C>
void launch_FixedAlgKernel(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params) {
    FixedAlgKernel<A, B, C><<<grid_size, block_size, 0, stream>>>(params);
}

// explicit function template instantiations for required datatypes
template void launch_MemsetKernel(int grid_size, int block_size, cudaStream_t stream, cuComplex* ptr, cuComplex value, int num_samples);

// fixed algorithm explicit function template instantiations - all combinations
template void launch_FixedAlgKernel<false, false, false>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<false, false,  true>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<false, true,  false>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<false, true,   true>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<true,  false, false>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<true,  false,  true>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<true,  true,  false>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);
template void launch_FixedAlgKernel<true,  true,   true>(int grid_size, int block_size, cudaStream_t stream, FixedAlgKernelParams params);

void fixedAlg_updateConstantMemory(float* src_ptr, size_t num_bytes) {
    fixedAlg_updateConstantMemory_internal(src_ptr, num_bytes);
}