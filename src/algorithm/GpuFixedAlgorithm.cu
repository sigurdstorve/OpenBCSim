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
#include "common_utils.hpp" // for compute_num_rf_samples

namespace bcsim {


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
                               float* res) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    float3 point = make_float3(point_xs[global_idx], point_ys[global_idx], point_zs[global_idx]) - origin;
    
    // compute dot products
    const auto radial_dist  = dot(point, rad_dir);
    const auto lateral_dist = dot(point, lat_dir);
    const auto elev_dist    = dot(point, ele_dir);


    const float two_sigma_lateral_squared     = 2.0f*sigma_lateral*sigma_lateral;
    const float two_sigma_elevational_squared = 2.0f*sigma_elevational*sigma_elevational; 
    const float weight = expf(-(lateral_dist*lateral_dist/two_sigma_lateral_squared + elev_dist*elev_dist/two_sigma_elevational_squared));

    const int radial_index = static_cast<int>(fs_hertz*2.0f*radial_dist/sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < num_time_samples) {
        //res[radial_index] += weight;
        atomicAdd(res+radial_index, weight*point_as[global_idx]);
    }
}


GpuFixedAlgorithm::GpuFixedAlgorithm()
{
}

void GpuFixedAlgorithm::projection_kernel(int stream_no, const Scanline& scanline) {
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

    dim3 grid_size(m_num_scatterers/m_param_threads_per_block, 1, 1);
    dim3 block_size(m_param_threads_per_block, 1, 1);
    
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
                                                             m_beam_profile->getSigmaLateral(),
                                                             m_beam_profile->getSigmaElevational(),
                                                             m_sound_speed,
                                                             m_device_time_proj[stream_no]->data());
    
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