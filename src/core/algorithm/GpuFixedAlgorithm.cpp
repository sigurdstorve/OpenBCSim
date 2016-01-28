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
#ifdef BCSIM_ENABLE_CUDA
#include <cuda.h>
#include "GpuFixedAlgorithm.hpp"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "cuda_kernels_c_interface.h"

namespace bcsim {

GpuFixedAlgorithm::GpuFixedAlgorithm()
{
}

void GpuFixedAlgorithm::projection_kernel(int stream_no, const Scanline& scanline, int num_blocks) {
    auto cur_stream = m_stream_wrappers[stream_no]->get();

    //dim3 grid_size(num_blocks, 1, 1);
    //dim3 block_size(m_param_threads_per_block, 1, 1);

    // prepare struct with parameters
    FixedAlgKernelParams params;
    params.point_xs          = m_device_point_xs->data();
    params.point_ys          = m_device_point_ys->data();
    params.point_zs          = m_device_point_zs->data();
    params.point_as          = m_device_point_as->data();
    params.rad_dir           = to_float3(scanline.get_direction());
    params.lat_dir           = to_float3(scanline.get_lateral_dir());
    params.ele_dir           = to_float3(scanline.get_elevational_dir());
    params.origin            = to_float3(scanline.get_origin());
    params.fs_hertz          = m_excitation.sampling_frequency;
    params.num_time_samples  = m_num_time_samples;
    params.sigma_lateral     = m_analytical_sigma_lat;
    params.sigma_elevational = m_analytical_sigma_ele;
    params.sound_speed       = m_param_sound_speed;
    params.res               = m_device_time_proj[stream_no]->data();
    params.demod_freq        = m_excitation.demod_freq;
    params.num_scatterers    = m_num_scatterers;
    params.lut_tex           = m_device_beam_profile->get();
    params.lut_r_min         = m_lut_r_min;
    params.lut_r_max         = m_lut_r_max;
    params.lut_l_min         = m_lut_l_min;
    params.lut_l_max         = m_lut_l_max;
    params.lut_e_min         = m_lut_e_min;
    params.lut_e_max         = m_lut_e_max;

    // map beam profile type to boolean flag
    bool use_lut;
    switch(m_cur_beam_profile_type) {
    case BeamProfileType::ANALYTICAL:
        use_lut = false;
        break;
    case BeamProfileType::LOOKUP:
        use_lut = true;
        break;
    default:
        throw std::logic_error("unknown beam profile type");
    }

    if (!m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<false, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<false, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<false, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<false, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<true, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<true, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<true, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<true, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else {
        throw std::logic_error("this should never happen");
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
#endif  // BCSIM_ENABLE_CUDA
