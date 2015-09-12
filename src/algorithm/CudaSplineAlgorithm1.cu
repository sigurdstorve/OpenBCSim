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

#include <iostream>
#include <vector>
#include <cuda.h>
#include <device_launch_parameters.h>
#include "cuda_helpers.h"
#include "bspline.hpp"
#include "LibBCSim.hpp"
#include "CudaSplineAlgorithm1.cuh"

#define MAX_CS 20

__constant__ float eval_basis[MAX_CS];

__global__ void RenderSplineKernel(const float* control_xs,
                                   const float* control_ys,
                                   const float* control_zs,
                                   const float* control_as,
                                   float* rendered_xs,
                                   float* rendered_ys,
                                   float* rendered_zs,
                                   float* rendered_as,
                                   int NUM_CS,
                                   int NUM_SPLINES) {

    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    float rendered_a = 0.0f;
    for (int i = 0; i < NUM_CS; i++) {
        rendered_x += control_xs[NUM_SPLINES*i + idx]*eval_basis[i];
        rendered_y += control_ys[NUM_SPLINES*i + idx]*eval_basis[i];
        rendered_z += control_zs[NUM_SPLINES*i + idx]*eval_basis[i];
        rendered_a += control_as[NUM_SPLINES*i + idx]*eval_basis[i];
    }

    // write result to memory
    rendered_xs[idx] = rendered_x;
    rendered_ys[idx] = rendered_y;
    rendered_zs[idx] = rendered_z;
    rendered_as[idx] = rendered_a;
}


namespace bcsim {

CudaSplineAlgorithm1::CudaSplineAlgorithm1() {
    m_fixed_alg = std::shared_ptr<CudaFixedAlgorithm>(new CudaFixedAlgorithm);
}

void CudaSplineAlgorithm1::set_scatterers(Scatterers::s_ptr new_scatterers) {
    auto scatterers = std::dynamic_pointer_cast<SplineScatterers>(new_scatterers);
    if (!scatterers) {
        throw std::runtime_error("Cast to SplineScatterers failed!");
    }

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
    m_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));
    m_control_as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(cs_num_bytes));

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
    cudaErrorCheck( cudaMemcpy(m_control_xs->data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_ys->data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_zs->data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_as->data(), host_control_as.data(), cs_num_bytes, cudaMemcpyHostToDevice) );


    // device memory to hold x, y, z, a components of rendered splines
    size_t rendered_num_bytes = m_num_splines*sizeof(float);
    m_fixed_alg->m_device_point_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_device_point_as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(rendered_num_bytes));
    m_fixed_alg->m_num_scatterers = m_num_splines;

    // Store the knot vector.
    m_common_knots = scatterers->knot_vector;
}

void CudaSplineAlgorithm1::simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines) {
    m_fixed_alg->simulate_lines(rf_lines);
}


void CudaSplineAlgorithm1::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {

    // For now it is a requirement that all lines in the scan sequence have the same
    // timestamp. This limitation will be removed in the future.
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



    // evaluate the basis functions and upload to constant memory.
    std::vector<float> host_basis_functions(m_num_cs);
    for (int i = 0; i < m_num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, m_spline_degree, PARAMETER_VAL, m_common_knots);
    }
    cudaErrorCheck( cudaMemcpyToSymbol(eval_basis, host_basis_functions.data(), m_num_cs*sizeof(float)) );
    
    //EventTimerRAII event_timer;

    // TODO: Handle non-power-of-<num_threads>
    int num_threads = 128;
    dim3 grid_size(m_num_splines/num_threads, 1, 1);
    dim3 block_size(num_threads, 1, 1);
    
    //event_timer.restart();
    RenderSplineKernel<<<grid_size, block_size>>>(m_control_xs->data(),
                                                  m_control_ys->data(),
                                                  m_control_zs->data(),
                                                  m_control_as->data(),
                                                  m_fixed_alg->m_device_point_xs->data(),
                                                  m_fixed_alg->m_device_point_ys->data(),
                                                  m_fixed_alg->m_device_point_zs->data(),
                                                  m_fixed_alg->m_device_point_as->data(),
                                                  m_num_cs,
                                                  m_num_splines);
    cudaErrorCheck( cudaDeviceSynchronize() );
    //auto ms = event_timer.stop();
    //std::cout << "set_scan_sequence(): rendering spline scatterers took " << ms << " millisec.\n";
}

bool CudaSplineAlgorithm1::has_equal_timestamps(ScanSequence::s_ptr scan_seq, double tol) {
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