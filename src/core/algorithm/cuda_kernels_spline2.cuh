#pragma once
#include "cuda_kernels_c_interface.h"

void splineAlg2_updateConstantMemory_internal(float* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
