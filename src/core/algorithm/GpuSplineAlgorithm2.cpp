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
#include <tuple>
#include <cuda.h>
#include "GpuSplineAlgorithm2.hpp"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "../bspline.hpp"
#include "common_utils.hpp"
#include "common_definitions.h" // for MAX_NUM_CUDA_STREAMS and MAX_SPLINE_DEGREE
#include "cuda_kernels_c_interface.h" // for SplineAlgKernelParams

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

void GpuSplineAlgorithm2::projection_kernel(int stream_no, const Scanline& scanline, int num_blocks) {
    auto cur_stream = m_stream_wrappers[stream_no]->get();
            
    // evaluate the basis functions and upload to constant memory.
    const auto num_nonzero = m_spline_degree+1;
    size_t eval_basis_offset_elements = num_nonzero*stream_no;
    std::vector<float> host_basis_functions(m_num_cs);
    for (int i = 0; i < m_num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, m_spline_degree, scanline.get_timestamp(), m_common_knots);
    }

    //dim3 grid_size(num_blocks, 1, 1);
    //dim3 block_size(m_param_threads_per_block, 1, 1);

    // TODO: Is it neccessary to have both m_num_splines AND m_num_scatterers? They
    // are equal...

    // compute sum limits (inclusive)
    int cs_idx_start, cs_idx_end;
    std::tie(cs_idx_start, cs_idx_end) = bspline_storve::get_lower_upper_inds(m_common_knots,
                                                                              scanline.get_timestamp(),
                                                                              m_spline_degree);
    if (!sanity_check_spline_lower_upper_bound(host_basis_functions, cs_idx_start, cs_idx_end)) {
        throw std::runtime_error("b-spline basis bounds failed sanity check");
    }
    if (cs_idx_end-cs_idx_start+1 != num_nonzero) throw std::logic_error("illegal number of non-zero basis functions");

    if(!splineAlg2_updateConstantMemory(host_basis_functions.data() + cs_idx_start,
                                        num_nonzero*sizeof(float),
                                        eval_basis_offset_elements*sizeof(float),
                                        cudaMemcpyHostToDevice,
                                        cur_stream))
    {
        throw std::runtime_error("Failed to copy to symbol memory");
    }

    // prepare a struct of arguments
    SplineAlgKernelParams params;
    params.control_xs                 = m_device_control_xs->data();
    params.control_ys                 = m_device_control_ys->data();
    params.control_zs                 = m_device_control_zs->data();
    params.control_as                 = m_device_control_as->data();
    params.rad_dir                    = to_float3(scanline.get_direction());
    params.lat_dir                    = to_float3(scanline.get_lateral_dir());
    params.ele_dir                    = to_float3(scanline.get_elevational_dir());
    params.origin                     = to_float3(scanline.get_origin());
    params.fs_hertz                   = m_excitation.sampling_frequency;
    params.num_time_samples           = static_cast<int>(m_num_time_samples);
    params.sigma_lateral              = m_analytical_sigma_lat;
    params.sigma_elevational          = m_analytical_sigma_ele;
    params.sound_speed                = m_param_sound_speed;
    params.cs_idx_start               = cs_idx_start;
    params.cs_idx_end                 = cs_idx_end;
    params.NUM_SPLINES                = m_num_splines;
    params.res                        = m_device_time_proj[stream_no]->data();
    params.eval_basis_offset_elements = eval_basis_offset_elements;
    params.demod_freq                 = m_excitation.demod_freq;
    params.lut_tex                    = m_device_beam_profile->get();
    params.lut.r_min                  = m_lut_r_min;
    params.lut.r_max                  = m_lut_r_max;
    params.lut.l_min                  = m_lut_l_min;
    params.lut.l_max                  = m_lut_l_max;
    params.lut.e_min                  = m_lut_e_min;
    params.lut.e_max                  = m_lut_e_max;

    // map lut type to a boolean flag
    bool use_lut;
    switch (m_cur_beam_profile_type) {
    case BeamProfileType::ANALYTICAL:
        use_lut = false;
        break;
    case BeamProfileType::LOOKUP:
        use_lut = true;
        break;
    default:
        throw std::logic_error("GpuSplineAlgorithm2: unknown beam profile type");
    }
    if (!m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_SplineAlgKernel<false, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_SplineAlgKernel<false, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_SplineAlgKernel<false, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_SplineAlgKernel<false, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_SplineAlgKernel<true, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_SplineAlgKernel<true, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_SplineAlgKernel<true, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_SplineAlgKernel<true, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else {
        throw std::logic_error("this should never happen");
    }
}

void GpuSplineAlgorithm2::copy_scatterers_to_device(SplineScatterers::s_ptr scatterers) {
    m_can_change_cuda_device = false;
    
    m_num_splines = scatterers->num_scatterers();
    if (m_num_splines <= 0) {
        throw std::runtime_error("No scatterers");
    }
    m_spline_degree = scatterers->spline_degree;
    m_num_cs = scatterers->get_num_control_points();

    if (m_spline_degree > MAX_SPLINE_DEGREE) {
        throw std::runtime_error("maximum spline degree supported is " + std::to_string(MAX_SPLINE_DEGREE));
    }

    std::cout << "Num spline scatterers: " << m_num_splines << std::endl;
    std::cout << "Allocating memory on host for reorganizing spline data\n";


    // device memory to hold x, y, z components of all spline control points and amplitudes of all splines.
    const size_t total_num_cs = m_num_splines*m_num_cs;
    const size_t cs_num_bytes = total_num_cs*sizeof(float);
    const size_t amplitudes_num_bytes = m_num_splines*sizeof(float);
    m_device_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_device_control_as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(amplitudes_num_bytes));
        
    // store the control points with correct memory layout of the host
    std::vector<float> host_control_xs(total_num_cs);
    std::vector<float> host_control_ys(total_num_cs);
    std::vector<float> host_control_zs(total_num_cs);
    std::vector<float> host_control_as(m_num_splines); // only one amplitude for each scatterer.

    for (size_t spline_no = 0; spline_no < m_num_splines; spline_no++) {
        host_control_as[spline_no] = scatterers->amplitudes[spline_no];
        for (size_t i = 0; i < m_num_cs; i++) {
            const size_t offset = spline_no + i*m_num_splines;
            host_control_xs[offset] = scatterers->control_points[spline_no][i].x;
            host_control_ys[offset] = scatterers->control_points[spline_no][i].y;
            host_control_zs[offset] = scatterers->control_points[spline_no][i].z;
        }
    }
    
    // copy control points to GPU memory.
    cudaErrorCheck( cudaMemcpy(m_device_control_xs->data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_device_control_ys->data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_device_control_zs->data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    
    // copy amplitudes to GPU memory.
    cudaErrorCheck( cudaMemcpy(m_device_control_as->data(), host_control_as.data(), amplitudes_num_bytes, cudaMemcpyHostToDevice) );

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
#endif  // BCSIM_ENABLE_CUDA
