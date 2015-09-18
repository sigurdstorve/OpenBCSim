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
#include "GpuSplineAlgorithm2.cuh"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "device_launch_parameters.h" // for removing annoying MSVC intellisense error messages
#include "bspline.hpp"
#include "gpu_alg_common.cuh" // for misc. CUDA kernels
#include <math_functions.h> // for copysignf()

// maximum number of spline control points for each scatterer
#define MAX_CS 20
// the maximum number of CUDA streams that can be used when simulating RF lines
#define MAX_NUM_CUDA_STREAMS 2


__constant__ float eval_basis[MAX_CS*MAX_NUM_CUDA_STREAMS];

__global__ void SplineAlgKernel(float* control_xs,
                                float* control_ys,
                                float* control_zs,
                                float* control_as,
                                float3 rad_dir,
                                float3 lat_dir,
                                float3 ele_dir,
                                float3 origin,
                                float  fs_hertz,
                                int    num_time_samples,
                                float  sigma_lateral,
                                float  sigma_elevational,
                                float  sound_speed,
                                int    NUM_CS,
                                int    NUM_SPLINES,
                                float* res,
                                size_t eval_basis_offset_elements) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // step 1: evaluate spline
    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    float rendered_a = 0.0f;
    for (int i = 0; i < NUM_CS; i++) {
        size_t eval_basis_i = i + eval_basis_offset_elements;
        rendered_x += control_xs[NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i];
        rendered_y += control_ys[NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i];
        rendered_z += control_zs[NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i];
        rendered_a += control_as[NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i];
    }

    // step 2: compute projections
    float3 point = make_float3(rendered_x, rendered_y, rendered_z) - origin;
    
    // compute dot products
    auto radial_dist  = dot(point, rad_dir);
    const auto lateral_dist = dot(point, lat_dir);
    const auto elev_dist    = dot(point, ele_dir);

    // Use "arc projection" in the radial direction: use length of vector from
    // beam's origin to the scatterer with the same sign as the projection onto
    // the line.
    radial_dist = copysignf(sqrtf(dot(point,point)), radial_dist);

    // compute weight
    const float two_sigma_lateral_squared     = 2.0f*sigma_lateral*sigma_lateral;
    const float two_sigma_elevational_squared = 2.0f*sigma_elevational*sigma_elevational; 
    const float weight = expf(-(lateral_dist*lateral_dist/two_sigma_lateral_squared + elev_dist*elev_dist/two_sigma_elevational_squared));

    const int radial_index = static_cast<int>(fs_hertz*2.0f*radial_dist/sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < num_time_samples) {
        //res[radial_index] += weight;
        atomicAdd(res+radial_index, weight*rendered_a);
    }
}


namespace bcsim {

GpuSplineAlgorithm2::GpuSplineAlgorithm2()
{
}

void GpuSplineAlgorithm2::set_parameter(const std::string& key, const std::string& value) {
    if (key == "cuda_streams") {
        const auto old_value = m_param_num_cuda_streams;
        if (m_param_num_cuda_streams > MAX_NUM_CUDA_STREAMS) {
            // reset to old value if new value is invalid
            m_param_num_cuda_streams = old_value;
            throw std::runtime_error("number of CUDA streams exceeds MAX_NUM_CUDA_STREAMS");
        }
    } else {
        GpuBaseAlgorithm::set_parameter(key, value);
    }
}

void GpuSplineAlgorithm2::projection_kernel(int stream_no, const Scanline& scanline) {
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
        
    // evaluate the basis functions and upload to constant memory.
    size_t eval_basis_offset_elements = m_num_cs*stream_no;
    std::vector<float> host_basis_functions(m_num_cs);
    for (int i = 0; i < m_num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, m_spline_degree, scanline.get_timestamp(), m_common_knots);
    }
    cudaErrorCheck(cudaMemcpyToSymbolAsync(eval_basis,
                                            host_basis_functions.data(),
                                            m_num_cs*sizeof(float),
                                            eval_basis_offset_elements*sizeof(float),
                                            cudaMemcpyHostToDevice,
                                            cur_stream));

    dim3 grid_size(m_num_scatterers/m_param_threads_per_block, 1, 1);
    dim3 block_size(m_param_threads_per_block, 1, 1);
    
    // do the time-projections
    SplineAlgKernel<<<grid_size, block_size, 0, cur_stream>>>(m_device_control_xs->data(),
                                                              m_device_control_ys->data(),
                                                              m_device_control_zs->data(),
                                                              m_device_control_as->data(),
                                                              rad_dir,
                                                              lat_dir,
                                                              ele_dir,
                                                              origin,
                                                              m_excitation.sampling_frequency,
                                                              m_num_time_samples,
                                                              m_beam_profile->getSigmaLateral(),
                                                              m_beam_profile->getSigmaElevational(),
                                                              m_param_sound_speed,
                                                              m_num_cs,
                                                              m_num_splines,
                                                              m_device_time_proj[stream_no]->data(),
                                                              eval_basis_offset_elements);

}

void GpuSplineAlgorithm2::copy_scatterers_to_device(SplineScatterers::s_ptr scatterers) {
    m_can_change_cuda_device = false;
    
    m_num_splines = scatterers->num_scatterers();
    if (m_num_splines <= 0) {
        throw std::runtime_error("No scatterers");
    }
    m_spline_degree = scatterers->spline_degree;
    m_num_cs = scatterers->nodes[0].size();

    if (m_num_cs > MAX_CS) {
        throw std::runtime_error("Too many control points in spline");
    }

    std::cout << "Num spline scatterers: " << m_num_splines << std::endl;
    std::cout << "Allocating memory on host for reorganizing spline data\n";


    // device memory to hold x, y, z components of all spline control points
    const size_t total_num_cs = m_num_splines*m_num_cs;
    const size_t cs_num_bytes = total_num_cs*sizeof(float);
    m_device_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
        
    // store the control points with correct memory layout of the host
    std::vector<float> host_control_xs(total_num_cs);
    std::vector<float> host_control_ys(total_num_cs);
    std::vector<float> host_control_zs(total_num_cs);
    std::vector<float> host_control_as(total_num_cs);

    for (size_t spline_no = 0; spline_no < m_num_splines; spline_no++) {
        for (size_t i = 0; i < m_num_cs; i++) {
            const size_t offset = spline_no + i*m_num_splines;
            host_control_xs[offset] = scatterers->nodes[spline_no][i].pos.x;
            host_control_ys[offset] = scatterers->nodes[spline_no][i].pos.y;
            host_control_zs[offset] = scatterers->nodes[spline_no][i].pos.z;
            host_control_as[offset] = scatterers->nodes[spline_no][i].amplitude;
        }
    }
    
    // copy control points to GPU memory.
    cudaErrorCheck( cudaMemcpy(m_device_control_xs->data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_device_control_ys->data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_device_control_zs->data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_device_control_as->data(), host_control_as.data(), cs_num_bytes, cudaMemcpyHostToDevice) );

    // Store the knot vector.
    m_common_knots = scatterers->knot_vector;
    
}

void GpuSplineAlgorithm2::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_can_change_cuda_device = false;
    
    m_num_scatterers = new_scatterers->num_scatterers();
        
    auto spline_scatterers = std::dynamic_pointer_cast<SplineScatterers>(new_scatterers);
    if (!spline_scatterers) {
        throw std::runtime_error("Cast failed");
    }
    copy_scatterers_to_device(spline_scatterers);
}


}   // end namespace
