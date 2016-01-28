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
#include <iostream>
#include <vector>
#include <tuple>
#include <cuda.h>
#include "cuda_helpers.h"
#include "../bspline.hpp"
#include "../LibBCSim.hpp"
#include "GpuSplineAlgorithm1.hpp"
#include "common_utils.hpp"
#include "common_definitions.h" // for MAX_SPLINE_DEGREE
#include "cuda_kernels_c_interface.h"

namespace bcsim {

GpuSplineAlgorithm1::GpuSplineAlgorithm1() {
    m_fixed_alg = std::shared_ptr<GpuFixedAlgorithm>(new GpuFixedAlgorithm);
}

void GpuSplineAlgorithm1::set_scatterers(Scatterers::s_ptr new_scatterers) {
    auto scatterers = std::dynamic_pointer_cast<SplineScatterers>(new_scatterers);
    if (!scatterers) {
        throw std::runtime_error("Cast to SplineScatterers failed!");
    }

    m_num_splines = scatterers->num_scatterers();
    if (m_num_splines <= 0) {
        throw std::runtime_error("No scatterers");
    }
    m_spline_degree = scatterers->spline_degree;

    if (m_spline_degree > MAX_SPLINE_DEGREE) {
        throw std::runtime_error("maximum spline degree supported is " + std::to_string(MAX_SPLINE_DEGREE));
    }

    m_num_cs = scatterers->get_num_control_points();
    std::cout << "Num spline scatterers: " << m_num_splines << std::endl;
    std::cout << "Allocating memory on host for reorganizing spline data\n";


    // device memory to hold x, y, z components of all spline control points
    const size_t total_num_cs = m_num_splines*m_num_cs;
    const size_t cs_num_bytes = total_num_cs*sizeof(float);
    m_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));

    // store the control points with correct memory layout of the host
    std::vector<float> host_control_xs(total_num_cs);
    std::vector<float> host_control_ys(total_num_cs);
    std::vector<float> host_control_zs(total_num_cs);
    std::vector<float> host_control_as(m_num_splines);

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
    cudaErrorCheck( cudaMemcpy(m_control_xs->data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_ys->data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_zs->data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );

    // device memory to hold x, y, z, a components of rendered splines
    size_t rendered_num_bytes = m_num_splines*sizeof(float);
    m_fixed_alg->m_device_point_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_num_scatterers = m_num_splines;

    // copy amplitudes directly from host memory.
    cudaErrorCheck( cudaMemcpy(m_fixed_alg->m_device_point_as->data(), host_control_as.data(), rendered_num_bytes, cudaMemcpyHostToDevice) );

    // Store the knot vector.
    m_common_knots = scatterers->knot_vector;
}

void GpuSplineAlgorithm1::simulate_lines(std::vector<std::vector<std::complex<float>> >&  /*out*/ rf_lines) {
    m_fixed_alg->simulate_lines(rf_lines);
}


void GpuSplineAlgorithm1::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    //EventTimerRAII event_timer;
    //event_timer.restart();

    // all lines in the scan sequence have the same timestamp
    if (!has_equal_timestamps(new_scan_sequence)) {
        throw std::runtime_error("scan sequences must currently have equal timestamps");
    }

    m_fixed_alg->set_scan_sequence(new_scan_sequence);

    // Ensure that set_scatterers() has been called first
    if (m_common_knots.size() == 0) {
        throw std::runtime_error("set_scatterers() must be called before set_scan_sequence");
    }

    const auto num_lines = new_scan_sequence->get_num_lines();
    if (num_lines <= 0) {
        throw std::runtime_error("No scanlines");
    }
    // HACK: using parameter value from first scanline
    const float PARAMETER_VAL = new_scan_sequence->get_scanline(0).get_timestamp();



    // evaluate the basis functions and upload to constant memory - always at max degree+1
    // basis functions that are non-zero at any parameter value.
    int cs_idx_start, cs_idx_end;
    std::tie(cs_idx_start, cs_idx_end) = bspline_storve::get_lower_upper_inds(m_common_knots,
                                                                              PARAMETER_VAL,
                                                                              m_spline_degree);
    const auto num_nonzero = cs_idx_end-cs_idx_start+1;
    if (num_nonzero != m_spline_degree+1) throw std::logic_error("illegal number of non-zero basis functions");

    // evaluate all basis functions since it will be checked that the ones supposed to
    // be zero in fact are zero.
    std::vector<float> host_basis_functions(m_num_cs); // TODO: move to set_scatterers()?
    for (int i = 0; i < m_num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, m_spline_degree, PARAMETER_VAL, m_common_knots);
    }
    
    if (!sanity_check_spline_lower_upper_bound(host_basis_functions, cs_idx_start, cs_idx_end)) {
        throw std::runtime_error("b-spline basis bounds failed sanity check");
    }
    
    // only copy the non-zero-basis functions
    const auto src_ptr = host_basis_functions.data() + cs_idx_start;
    fixedAlg_updateConstantMemory(src_ptr, num_nonzero*sizeof(float));
    
    int num_threads = 128;
    int num_blocks = round_up_div(m_num_splines, num_threads);
    dim3 grid_size(num_blocks, 1, 1);
    dim3 block_size(num_threads, 1, 1);
    
    // TODO: UPDATE CUDA
    /*
    RenderSplineKernel<<<grid_size, block_size>>>(m_control_xs->data(),
                                                  m_control_ys->data(),
                                                  m_control_zs->data(),
                                                  m_fixed_alg->m_device_point_xs->data(),
                                                  m_fixed_alg->m_device_point_ys->data(),
                                                  m_fixed_alg->m_device_point_zs->data(),
                                                  cs_idx_start,
                                                  cs_idx_end,
                                                  m_num_splines);
    */
    cudaErrorCheck( cudaDeviceSynchronize() );
    //auto ms = event_timer.stop();
    //std::cout << "GPU spline alg.1 : set_scan_sequence(): rendering spline scatterers took " << ms << " millisec.\n";
}

bool GpuSplineAlgorithm1::has_equal_timestamps(ScanSequence::s_ptr scan_seq, double tol) {
    const auto num_lines = scan_seq->get_num_lines();
    if (num_lines <= 1) {
        return true;
    }
    const auto common_time = scan_seq->get_scanline(0).get_timestamp();
    for (int i = 1; i < num_lines; i++) {
        const auto timestamp = scan_seq->get_scanline(i).get_timestamp();
        if (std::abs(timestamp-common_time) > tol) {
            return false;
        }
    }
    return true;
}

}   // end namespace
#endif  // BCSIM_ENABLE_CUDA
