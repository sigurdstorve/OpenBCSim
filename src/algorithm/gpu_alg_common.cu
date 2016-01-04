/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cuda.h>
#include <cufft.h>
#include <math_functions.h> // for sincosf()
#include "cuda_helpers.h"
#include "device_launch_parameters.h" // for removing annoying MSVC intellisense error messages
#include "gpu_alg_common.cuh"

__global__ void MultiplyFftKernel(cufftComplex* time_proj_fft, const cufftComplex* filter_fft, int num_samples) {
    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx < num_samples) {
        cufftComplex a = time_proj_fft[global_idx];
        cufftComplex b = filter_fft[global_idx];
        time_proj_fft[global_idx] = make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); 
    }
}

__global__ void ScaleSignalKernel(cufftComplex* signal, float factor, int num_samples) {
    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx < num_samples) {
        cufftComplex c     = signal[global_idx];
        signal[global_idx] = make_float2(c.x*factor, c.y*factor);
    }
}

__global__ void DemodulateKernel(cuComplex* signal, float w, int num_samples) {
    const auto global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx < num_samples) {
        // exp(-i*w*n) = cos(w*n) - i*sin(w*n)
        float sin_value, cos_value;
        sincosf(w*global_idx, &sin_value, &cos_value);
        const auto c = make_cuComplex(cos_value, -sin_value);

        signal[global_idx] = ComplexMul(signal[global_idx], c);
    }
}
