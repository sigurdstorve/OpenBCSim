#pragma once
#include "cuda_kernels_c_interface.h"

void splineAlg2_updateConstantMemory_internal(float* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);

template <bool use_arc_projection, bool use_phase_delay, bool use_lut>
__global__ void SplineAlgKernel(SplineAlgKernelParams params);
