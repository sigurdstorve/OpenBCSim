#include "cuda_kernels_c_interface.h"
#include "cuda_kernels_common.cuh"      // for common kernels
#include "cuda_kernels_fixed.cuh"       // for FixedAlgKernel
#include "cuda_kernels_spline1.cuh"     // for fixedAlg_updateConstantMemory_internal
#include "cuda_kernels_spline2.cuh"     // for splineAlg2_updateConstantMemory_internal

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

bool fixedAlg_updateConstantMemory(float* src_ptr, size_t num_bytes) {
    return fixedAlg_updateConstantMemory_internal(src_ptr, num_bytes);
}

void launch_RenderSplineKernel(int grid_size, int block_size, cudaStream_t stream,
                               const float* control_xs,
                               const float* control_ys,
                               const float* control_zs,
                               float* rendered_xs,
                               float* rendered_ys,
                               float* rendered_zs,
                               int cs_idx_start,
                               int cs_idx_end,
                               int NUM_SPLINES) {
    RenderSplineKernel<<<grid_size, block_size, 0, stream>>>(control_xs, control_ys, control_zs,
                                                             rendered_xs, rendered_ys, rendered_zs,
                                                             cs_idx_start, cs_idx_end, NUM_SPLINES);
}

void launch_SliceLookupTable(int grid_size0, int grid_size1, int block_size, cudaStream_t stream,
                             float3 origin,
                             float3 dir0,
                             float3 dir1,
                             float* output,
                             cudaTextureObject_t lut_tex) {
    dim3 grid_size(grid_size0, grid_size1, 1);
    SliceLookupTable<<<grid_size, block_size, 0, stream>>>(origin, dir0, dir1, output, lut_tex);
}

bool splineAlg2_updateConstantMemory(float* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    return splineAlg2_updateConstantMemory_internal(src, count, offset, kind, stream);
}

template <bool A, bool B, bool C>
void launch_SplineAlgKernel(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params) {
    SplineAlgKernel<A, B, C><<<grid_size, block_size, 0, stream>>>(params);
}

// spline algorithm2 explicit function template instantiations - all combinations
template void launch_SplineAlgKernel<false, false, false>(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<false, false, true >(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<false, true,  false>(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<false, true,  true >(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<true,  false, false>(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<true,  false, true >(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<true,  true,  false>(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
template void launch_SplineAlgKernel<true,  true,  true >(int grid_size, int block_size, cudaStream_t stream, SplineAlgKernelParams params);
