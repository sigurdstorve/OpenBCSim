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
#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cufft.h>
#include "cuda_kernels_c_interface.h"   // for struct LUTProfileGeometry

// initialize GPU memory with value
template <typename T>
__global__ void MemsetKernel(T* res, T value, int num_samples) {
    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx < num_samples) {
        res[global_idx] = value;
    }
}

// Compute projection weight from Gaussian analytical beam profile.
__device__ __inline__ float ComputeWeightAnalytical(float sigma_lateral,
                                                    float sigma_elevational,
                                                    float radial_dist,
                                                    float lateral_dist,
                                                    float elev_dist) {
    const float two_sigma_lateral_squared     = 2.0f*sigma_lateral*sigma_lateral;
    const float two_sigma_elevational_squared = 2.0f*sigma_elevational*sigma_elevational; 
    return expf(-(lateral_dist*lateral_dist/two_sigma_lateral_squared + elev_dist*elev_dist/two_sigma_elevational_squared));
}

// Compute projection weight from a 3D texture based beam profile.
__device__ __inline__ float ComputeWeightLUT(cudaTextureObject_t lut_tex,
                                             float radial_dist,
                                             float lateral_dist, 
                                             float elev_dist,
                                             LUTProfileGeometry lut_geo) {
    const auto r_normalized = (radial_dist-lut_geo.r_min)/(lut_geo.r_max-lut_geo.r_min);
    const auto l_normalized = (lateral_dist-lut_geo.l_min)/(lut_geo.l_max-lut_geo.l_min);
    const auto e_normalized = (elev_dist-lut_geo.e_min)/(lut_geo.e_max-lut_geo.e_min);
    return tex3D<float>(lut_tex, l_normalized, e_normalized, r_normalized);
}

// used to multiply the FFTs
__global__ void MultiplyFftKernel(cufftComplex* time_proj_fft, const cufftComplex* filter_fft, int num_samples);

// scale a signal (to avoid losing precision)
__global__ void ScaleSignalKernel(cufftComplex* signal, float factor, int num_samples);

// inplace IQ demodulation.
// normalized_angular_freq = 2*pi*f_demod, where f_demod in [0.0, 0.5]
__global__ void DemodulateKernel(cuComplex* signal, float normalized_angular_freq, int num_samples);

// add noise to a signal
__global__ void AddNoiseKernel(cuComplex* signal, cuComplex* noise, int num_samples);