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

struct FixedAlgKernelParams {
    float* point_xs;            // pointer to device memory x components
    float* point_ys;            // pointer to device memory y components
    float* point_zs;            // pointer to device memory z components
    float* point_as;            // pointer to device memory amplitudes
    float3 rad_dir;             // radial direction unit vector
    float3 lat_dir;             // lateral direction unit vector
    float3 ele_dir;             // elevational direction unit vector
    float3 origin;              // beam's origin
    float  fs_hertz;            // temporal sampling frequency in hertz
    int    num_time_samples;    // number of samples in time signal
    float  sigma_lateral;       // lateral beam width (for analyical beam profile)
    float  sigma_elevational;   // elevational beam width (for analytical beam profile)
    float  sound_speed;         // speed of sound in meters per second
    cuComplex* res;             // the output buffer (complex projected amplitudes)
    float  demod_freq;          // complex demodulation frequency.
    int    num_scatterers;      // number of scatterers
    cudaTextureObject_t lut_tex; // 3D texture object (for lookup-table beam profile)
    float lut_r_min;            // min. radial extent (for lookup-table beam profile)
    float lut_r_max;            // max. radial extent (for lookup-table beam profile)
    float lut_l_min;            // min. lateral extent (for lookup-table beam profile)
    float lut_l_max;            // max. lateral extent (for lookup-table beam profile)
    float lut_e_min;            // min. elevational extent (for lookup-table beam profile)
    float lut_e_max;            // max. elevational extent (for lookup-table beam profile)
    bool   use_arc_projection;  // TEMPORARY FLAG - WILL BE REPLACED WITH TEMPLATE PARAMETER
    bool   use_phase_delay;     // TEMPORARY FLAG - WILL BE REPLACED WITH TEMPLATE PARAMETER
    bool   use_lut;             // TEMPORARY FLAG - WILL BE REPLACED WITH TEMPLATE PARAMETER
};

__global__ void FixedAlgKernel(FixedAlgKernelParams params) {
    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx >= params.num_scatterers) {
        return;
    }

    const float3 point = make_float3(params.point_xs[global_idx], params.point_ys[global_idx], params.point_zs[global_idx]) - params.origin;
    
    // compute dot products
    auto radial_dist  = dot(point, params.rad_dir);
    const auto lateral_dist = dot(point, params.lat_dir);
    const auto elev_dist    = dot(point, params.ele_dir);

    if (params.use_arc_projection) {
        // Use "arc projection" in the radial direction: use length of vector from
        // beam's origin to the scatterer with the same sign as the projection onto
        // the line.
        radial_dist = copysignf(sqrtf(dot(point,point)), radial_dist);
    }

    float weight;
    if (params.use_lut) {
        // Compute weight from lookup-table and radial_dist, lateral_dist, and elev_dist
        weight = ComputeWeightLUT(params.lut_tex, radial_dist, lateral_dist, elev_dist,
                                  params.lut_r_min, params.lut_r_max, params.lut_l_min, params.lut_l_max, params.lut_e_min, params.lut_e_max);
    } else {
        weight = ComputeWeightAnalytical(params.sigma_lateral, params.sigma_elevational, radial_dist, lateral_dist, elev_dist);
    }

    const int radial_index = static_cast<int>(params.fs_hertz*2.0f*radial_dist/params.sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < params.num_time_samples) {
        //atomicAdd(res+radial_index, weight*point_as[global_idx]);
        if (params.use_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = params.fs_hertz*2.0f*radial_dist/params.sound_speed;
            const float ss_delay = (radial_index - true_index)/params.fs_hertz;
            const float complex_phase = 6.283185307179586*params.demod_freq*ss_delay;

            // exp(i*theta) = cos(theta) + i*sin(theta)
            float sin_value, cos_value;
            sincosf(complex_phase, &sin_value, &cos_value);

            const auto w = weight*params.point_as[global_idx];
            atomicAdd(&(params.res[radial_index].x), w*cos_value);
            atomicAdd(&(params.res[radial_index].y), w*sin_value);
        } else {
            atomicAdd(&(params.res[radial_index].x), weight*params.point_as[global_idx]);
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
    
    // prepare struct with parameters
    FixedAlgKernelParams params;
    params.point_xs          = m_device_point_xs->data();
    params.point_ys          = m_device_point_ys->data();
    params.point_zs          = m_device_point_zs->data();
    params.point_as          = m_device_point_as->data();
    params.rad_dir           = rad_dir;
    params.lat_dir           = lat_dir;
    params.ele_dir           = ele_dir;
    params.origin            = origin;
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

    // TEMP FLAGS - WILL BE REPLACED WITH TEMPLATE PARAMS
    params.use_arc_projection = m_param_use_arc_projection;
    params.use_phase_delay    = m_enable_phase_delay;
    params.use_lut            = use_lut;

    FixedAlgKernel<<<grid_size, block_size, 0, cur_stream>>>(params);
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
