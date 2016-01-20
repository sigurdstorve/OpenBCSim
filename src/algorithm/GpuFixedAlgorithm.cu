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
#include "GpuFixedAlgorithm.cuh"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "device_launch_parameters.h" // for removing annoying MSVC intellisense error messages
#include "gpu_alg_common.cuh" // for misc. CUDA kernels
#include <math_functions.h> // for copysignf

namespace bcsim {

// Gaussian analytical beam profile.
__device__ float ComputeWeightAnalytical(float sigma_lateral,
                                         float sigma_elevational,
                                         float radial_dist,
                                         float lateral_dist,
                                         float elev_dist) {
    const float two_sigma_lateral_squared     = 2.0f*sigma_lateral*sigma_lateral;
    const float two_sigma_elevational_squared = 2.0f*sigma_elevational*sigma_elevational; 
    return expf(-(lateral_dist*lateral_dist/two_sigma_lateral_squared + elev_dist*elev_dist/two_sigma_elevational_squared));
}

__global__ void FixedAlgKernel(float* point_xs,
                               float* point_ys,
                               float* point_zs,
                               float* point_as,
                               float3 rad_dir,
                               float3 lat_dir,
                               float3 ele_dir,
                               float3 origin,
                               float  fs_hertz,
                               int    num_time_samples,
                               float  sigma_lateral,
                               float  sigma_elevational,
                               float  sound_speed,
                               cuComplex* res,
                               bool   use_arc_projection,
                               int    num_scatterers,
                               bool   use_phase_delay,
                               float  demod_freq) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx >= num_scatterers) {
        return;
    }

    float3 point = make_float3(point_xs[global_idx], point_ys[global_idx], point_zs[global_idx]) - origin;
    
    // compute dot products
    auto radial_dist  = dot(point, rad_dir);
    const auto lateral_dist = dot(point, lat_dir);
    const auto elev_dist    = dot(point, ele_dir);

    if (use_arc_projection) {
        // Use "arc projection" in the radial direction: use length of vector from
        // beam's origin to the scatterer with the same sign as the projection onto
        // the line.
        radial_dist = copysignf(sqrtf(dot(point,point)), radial_dist);
    }

    const float weight = ComputeWeightAnalytical(sigma_lateral, sigma_elevational, radial_dist, lateral_dist, elev_dist);

    const int radial_index = static_cast<int>(fs_hertz*2.0f*radial_dist/sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < num_time_samples) {
        //atomicAdd(res+radial_index, weight*point_as[global_idx]);
        if (use_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = fs_hertz*2.0f*radial_dist/sound_speed;
            const float ss_delay = (radial_index - true_index)/fs_hertz;
            const float complex_phase = 6.283185307179586*demod_freq*ss_delay;

            // exp(i*theta) = cos(theta) + i*sin(theta)
            float sin_value, cos_value;
            sincosf(complex_phase, &sin_value, &cos_value);

            const auto w = weight*point_as[global_idx];
            atomicAdd(&(res[radial_index].x), w*cos_value);
            atomicAdd(&(res[radial_index].y), w*sin_value);
        } else {
            atomicAdd(&(res[radial_index].x), weight*point_as[global_idx]);
        }
    }
}

__global__ void FixedAlgKernel_LUT(float* point_xs,
                                   float* point_ys,
                                   float* point_zs,
                                   float* point_as,
                                   float3 rad_dir,
                                   float3 lat_dir,
                                   float3 ele_dir,
                                   float3 origin,
                                   float  fs_hertz,
                                   int    num_time_samples,
                                   float  sound_speed,
                                   cuComplex* res,
                                   bool   use_arc_projection,
                                   int    num_scatterers,
                                   bool   use_phase_delay,
                                   float  demod_freq,
                                   cudaTextureObject_t lut_tex) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx >= num_scatterers) {
        return;
    }

    float3 point = make_float3(point_xs[global_idx], point_ys[global_idx], point_zs[global_idx]) - origin;
    
    // compute dot products
    auto radial_dist  = dot(point, rad_dir);
    const auto lateral_dist = dot(point, lat_dir);
    const auto elev_dist    = dot(point, ele_dir);

    if (use_arc_projection) {
        // Use "arc projection" in the radial direction: use length of vector from
        // beam's origin to the scatterer with the same sign as the projection onto
        // the line.
        radial_dist = copysignf(sqrtf(dot(point,point)), radial_dist);
    }

    // TODO: Compute weight from lookup-table and radial_dist, lateral_dist, and elev_dist
    const auto lut_r_min = 0.0f; // HACK: this should be sent as parameters
    const auto lut_r_max = 0.12f;
    const auto lut_l_min = -2e-2f;
    const auto lut_l_max = 2e-2f;
    const auto lut_e_min = -2e-2f;
    const auto lut_e_max = 2e-2;
    
    const auto r_normalized = (radial_dist-lut_r_min)/(lut_r_max-lut_r_min);
    const auto l_normalized = (lateral_dist-lut_l_min)/(lut_l_max-lut_l_min);
    const auto e_normalized = (elev_dist-lut_e_min)/(lut_e_max-lut_e_min);
    const auto weight = tex3D<float>(lut_tex, l_normalized, e_normalized, r_normalized);

    const int radial_index = static_cast<int>(fs_hertz*2.0f*radial_dist/sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < num_time_samples) {
        //atomicAdd(res+radial_index, weight*point_as[global_idx]);
        if (use_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = fs_hertz*2.0f*radial_dist/sound_speed;
            const float ss_delay = (radial_index - true_index)/fs_hertz;
            const float complex_phase = 6.283185307179586*demod_freq*ss_delay;

            // exp(i*theta) = cos(theta) + i*sin(theta)
            float sin_value, cos_value;
            sincosf(complex_phase, &sin_value, &cos_value);

            const auto w = weight*point_as[global_idx];
            atomicAdd(&(res[radial_index].x), w*cos_value);
            atomicAdd(&(res[radial_index].y), w*sin_value);
        } else {
            atomicAdd(&(res[radial_index].x), weight*point_as[global_idx]);
        }
    }
}


GpuFixedAlgorithm::GpuFixedAlgorithm()
{
}

void GpuFixedAlgorithm::projection_kernel(int stream_no, const Scanline& scanline, int num_blocks) {
    auto cur_stream = m_stream_wrappers[stream_no]->get();

    // TODO: Move out conversion code            
    auto temp_rad_dir = scanline.get_direction();
    auto temp_lat_dir = scanline.get_lateral_dir();
    auto temp_ele_dir = scanline.get_elevational_dir();
    auto temp_origin  = scanline.get_origin();
    auto rad_dir      = make_float3(temp_rad_dir.x, temp_rad_dir.y, temp_rad_dir.z);
    auto lat_dir      = make_float3(temp_lat_dir.x, temp_lat_dir.y, temp_lat_dir.z);
    auto ele_dir      = make_float3(temp_ele_dir.x, temp_ele_dir.y, temp_ele_dir.z);
    auto origin       = make_float3(temp_origin.x, temp_origin.y, temp_origin.z);

    dim3 grid_size(num_blocks, 1, 1);
    dim3 block_size(m_param_threads_per_block, 1, 1);
    
    // Use casting of lookup_table to determine which kernel to call
    const auto gaussian_beam_profile = std::dynamic_pointer_cast<bcsim::GaussianBeamProfile>(m_beam_profile);
    const auto lut_beam_profile      = std::dynamic_pointer_cast<bcsim::LUTBeamProfile>(m_beam_profile);

    if (gaussian_beam_profile) {
        FixedAlgKernel<<<grid_size, block_size, 0, cur_stream>>>(m_device_point_xs->data(),
                                                                 m_device_point_ys->data(),
                                                                 m_device_point_zs->data(),
                                                                 m_device_point_as->data(),
                                                                 rad_dir,
                                                                 lat_dir,
                                                                 ele_dir,
                                                                 origin,
                                                                 m_excitation.sampling_frequency,
                                                                 m_num_time_samples,
                                                                 gaussian_beam_profile->getSigmaLateral(),
                                                                 gaussian_beam_profile->getSigmaElevational(),
                                                                 m_param_sound_speed,
                                                                 m_device_time_proj[stream_no]->data(),
                                                                 m_param_use_arc_projection,
                                                                 m_num_scatterers,
                                                                 m_enable_phase_delay,
                                                                 m_excitation.demod_freq);
    } else if (lut_beam_profile) {
        FixedAlgKernel_LUT<<<grid_size, block_size, 0, cur_stream>>>(m_device_point_xs->data(),
                                                                     m_device_point_ys->data(),
                                                                     m_device_point_zs->data(),
                                                                     m_device_point_as->data(),
                                                                     rad_dir,
                                                                     lat_dir,
                                                                     ele_dir,
                                                                     origin,
                                                                     m_excitation.sampling_frequency,
                                                                     m_num_time_samples,
                                                                     m_param_sound_speed,
                                                                     m_device_time_proj[stream_no]->data(),
                                                                     m_param_use_arc_projection,
                                                                     m_num_scatterers,
                                                                     m_enable_phase_delay,
                                                                     m_excitation.demod_freq,
                                                                     m_device_beam_profile->get()
                                                                     );
    }
}


void GpuFixedAlgorithm::copy_scatterers_to_device(FixedScatterers::s_ptr scatterers) {
    m_can_change_cuda_device = false;
    
    const size_t num_scatterers = scatterers->num_scatterers();
    size_t points_common_bytes = num_scatterers*sizeof(float);

    // temporary host memory for scatterer points
    HostPinnedBufferRAII<float> host_temp(points_common_bytes);

    // no point in reallocating if we already have allocated memory and the number of bytes
    // is the same.
    bool reallocate_device_mem = true;
    if (m_device_point_xs && m_device_point_ys && m_device_point_zs && m_device_point_as) {
        if (   (m_device_point_xs->get_num_bytes() == points_common_bytes)
            && (m_device_point_ys->get_num_bytes() == points_common_bytes)
            && (m_device_point_zs->get_num_bytes() == points_common_bytes)
            && (m_device_point_as->get_num_bytes() == points_common_bytes))
        {
            reallocate_device_mem = false;
        }
    }
    if (reallocate_device_mem) {
        m_device_point_xs = std::move(DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(points_common_bytes)));
        m_device_point_ys = std::move(DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(points_common_bytes)));
        m_device_point_zs = std::move(DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(points_common_bytes)));
        m_device_point_as = std::move(DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(points_common_bytes)));
    }

    // x values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = scatterers->scatterers[i].pos.x;
    }
    cudaErrorCheck( cudaMemcpy(m_device_point_xs->data(), host_temp.data(), points_common_bytes, cudaMemcpyHostToDevice) );

    // y values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = scatterers->scatterers[i].pos.y;
    }
    cudaErrorCheck( cudaMemcpy(m_device_point_ys->data(), host_temp.data(), points_common_bytes, cudaMemcpyHostToDevice) );

    // z values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = scatterers->scatterers[i].pos.z;
    }
    cudaErrorCheck( cudaMemcpy(m_device_point_zs->data(), host_temp.data(), points_common_bytes, cudaMemcpyHostToDevice) );

    // a values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = scatterers->scatterers[i].amplitude;
    }
    cudaErrorCheck( cudaMemcpy(m_device_point_as->data(), host_temp.data(), points_common_bytes, cudaMemcpyHostToDevice) );
}

void GpuFixedAlgorithm::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_can_change_cuda_device = false;
    m_num_scatterers = new_scatterers->num_scatterers();
        
    auto fixed_scatterers = std::dynamic_pointer_cast<FixedScatterers>(new_scatterers);
    if (!fixed_scatterers) {
        throw std::runtime_error("Cast failed");
    }
    copy_scatterers_to_device(fixed_scatterers);
}


}   // end namespace
