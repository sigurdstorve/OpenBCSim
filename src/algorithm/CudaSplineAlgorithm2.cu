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
#include <complex>
#include "CudaSplineAlgorithm2.cuh"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "device_launch_parameters.h" // for removing annoying MSVC intellisense error messages
#include "discrete_hilbert_mask.hpp"
#include "common_utils.hpp" // for compute_num_rf_samples
#include "bspline.hpp"
#include "gpu_alg_common.cuh" // for misc. CUDA kernels

#define MAX_CS 20

__constant__ float eval_basis[MAX_CS];

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
                                float* res) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // step 1: evaluate spline
    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    float rendered_a = 0.0f;
    for (int i = 0; i < NUM_CS; i++) {
        rendered_x += control_xs[NUM_SPLINES*i + global_idx]*eval_basis[i];
        rendered_y += control_ys[NUM_SPLINES*i + global_idx]*eval_basis[i];
        rendered_z += control_zs[NUM_SPLINES*i + global_idx]*eval_basis[i];
        rendered_a += control_as[NUM_SPLINES*i + global_idx]*eval_basis[i];
    }

    // step 2: compute projections
    float3 point = make_float3(rendered_x, rendered_y, rendered_z) - origin;
    
    // compute dot products
    const auto radial_dist  = dot(point, rad_dir);
    const auto lateral_dist = dot(point, lat_dir);
    const auto elev_dist    = dot(point, ele_dir);

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

CudaSplineAlgorithm2::CudaSplineAlgorithm2()
    : m_verbose(false),
      m_num_cuda_streams(2),
      m_num_time_samples(32768),  // TODO: remove this limitation
      m_num_beams_allocated(-1),
      m_beam_profile(nullptr)
{
    // create CUDA stream wrappers
    m_stream_wrappers.resize(m_num_cuda_streams);
    for (int i = 0; i < m_num_cuda_streams; i++) {
        m_stream_wrappers[i] = std::move(CudaStreamRAII::u_ptr(new CudaStreamRAII));
    }
    
    int device_count;
    cudaErrorCheck( cudaGetDeviceCount(&device_count) );
    std::cout << "CUDA device count: " << device_count << std::endl;
    
    for (int device_no = 0; device_no < device_count; device_no++) {
        cudaDeviceProp prop;
        cudaErrorCheck( cudaGetDeviceProperties(&prop, device_no) );
        std::cout << "\n\n=== Device " << device_no << ": " << prop.name << std::endl;
        std::cout << "totalGlobMem: " << prop.totalGlobalMem << std::endl;
        std::cout << "clockRate: " << prop.clockRate << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;
        std::cout << "multiProcessorCount: " << prop.multiProcessorCount << std::endl;
        std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "computeMode: " << prop.computeMode << std::endl;
        std::cout << "concurrentKernels: " << prop.concurrentKernels << std::endl;
        std::cout << "ECCEnabled: " << prop.ECCEnabled << std::endl;
        std::cout << "memoryBusWidth: " << prop.memoryBusWidth << std::endl;
    }

    std::cout << "For now using the first device. TODO: make changable\n";
    cudaErrorCheck( cudaSetDevice(0) );

}

void CudaSplineAlgorithm2::simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines) {

    auto num_lines      = m_scan_seq->get_num_lines();

    if (num_lines < 1) {
        throw std::runtime_error("No scanlines in scansequence");
    }

    if (m_beam_profile == nullptr) {
        throw std::runtime_error("No beam profile is set");
    }
    
    // evaluate the basis functions and upload to constant memory.
    // HACK: using parameter value from first scanline
    const float PARAMETER_VAL = m_scan_seq->get_scanline(0).get_timestamp();
    std::vector<float> host_basis_functions(m_num_cs);
    for (int i = 0; i < m_num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, m_spline_degree, PARAMETER_VAL, m_common_knots);
    }
    cudaErrorCheck( cudaMemcpyToSymbol(eval_basis, host_basis_functions.data(), m_num_cs*sizeof(float)) );
    
    
    int threads_pr_block = 128;
    dim3 grid_size(m_num_scatterers/threads_pr_block, 1, 1);
    dim3 block_size(threads_pr_block, 1, 1);

    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();

        if (m_verbose) {
            std::cout << "beam_no = " << beam_no << ", stream_no = " << stream_no << std::endl;
        }

        // TODO: Move out conversion code            
        auto scanline = m_scan_seq->get_scanline(beam_no);
        auto temp_rad_dir = scanline.get_direction();
        auto temp_lat_dir = scanline.get_lateral_dir();
        auto temp_ele_dir = scanline.get_elevational_dir();
        auto temp_origin  = scanline.get_origin();
        auto rad_dir      = make_float3(temp_rad_dir.x, temp_rad_dir.y, temp_rad_dir.z);
        auto lat_dir      = make_float3(temp_lat_dir.x, temp_lat_dir.y, temp_lat_dir.z);
        auto ele_dir      = make_float3(temp_ele_dir.x, temp_ele_dir.y, temp_ele_dir.z);
        auto origin       = make_float3(temp_origin.x, temp_origin.y, temp_origin.z);

        //std::cout << "origin: " << origin.x << " " << origin.y << " " << origin.z << std::endl;

        int threads_per_line = 128;
        // clear the time projection buffer the proper way (probably slightly slower than cudaMamSetAsync...)
        MemsetFloatKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_time_proj[stream_no]->data(),
                                                                                                    0.0f,
                                                                                                    m_num_time_samples);

        //if (beam_no==0) { dump_device_memory<float>(device_time_proj[stream_no]->data(), m_num_time_samples, "01_zeroed_rf_line_dump.txt"); }
        
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
                                                                  m_sim_params.sound_speed,
                                                                  m_num_cs,
                                                                  m_num_splines,
                                                                  m_device_time_proj[stream_no]->data());
        
            
        //if (beam_no==0) { dump_device_memory<float>(device_time_proj[stream_no]->data(), m_num_time_samples, "02_time_proj_dump.txt"); }


        // extend the real-valued time-projection signal to complex numbers with zero imaginary part.
        RealToComplexKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_time_proj[stream_no]->data(),
                                                                                                      m_device_rf_lines[stream_no]->data(),
                                                                                                      m_num_time_samples);
        //if (beam_no==0) { dump_device_memory<std::complex<float> >(reinterpret_cast<std::complex<float>*>(device_rf_lines[stream_no]->data()), m_num_time_samples, "03_complex_extension.txt"); }

        // in-place forward FFT            
        auto rf_ptr = m_device_rf_lines[stream_no]->data();
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_FORWARD) );

        // multiply with FFT of impulse response (can include Hilbert transform also)
        MultiplyFftKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                    m_device_excitation_fft->data(),
                                                                                                    m_num_time_samples);

        // in-place inverse FFT
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_INVERSE) );
            
        //if (beam_no==0) { dump_device_memory<std::complex<float> >(reinterpret_cast<std::complex<float>*>(rf_ptr), m_num_time_samples, "04_iq_line.txt"); }

        // envelope detection
        AbsComplexKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                   m_device_rf_lines_env[stream_no]->data(),
                                                                                                   m_num_time_samples);
            
        //if (beam_no==0) { dump_device_memory<float>(device_rf_lines_env[stream_no]->data(), m_num_time_samples, "05_rf_envelope.txt"); }
            
        // copy to host
        cudaErrorCheck( cudaMemcpyAsync(m_host_rf_lines[beam_no]->data(), m_device_rf_lines_env[stream_no]->data(), sizeof(float)*m_num_time_samples, cudaMemcpyDeviceToHost, cur_stream) ); 
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    // TODO: eliminate unneccessary data copying: it would e.g. be better to
    // only copy what is needed in the above kernel.
    
    const auto num_return_samples = compute_num_rf_samples(m_sim_params.sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);

    // compensate for delay
    const size_t start_idx = static_cast<size_t>(m_excitation.center_index);

    rf_lines.clear();
    std::vector<bc_float> temp_samples(num_return_samples);
    for (size_t line_no = 0; line_no < num_lines; line_no++) {
        for (size_t i = 0; i < num_return_samples; i++) {
            temp_samples[i] = m_host_rf_lines[line_no]->data()[i+start_idx];
        }
        rf_lines.push_back(temp_samples);
    }
}

void CudaSplineAlgorithm2::copy_scatterers_to_device(SplineScatterers::s_ptr scatterers) {

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

void CudaSplineAlgorithm2::set_excitation(const ExcitationSignal& new_excitation) {
    m_excitation = new_excitation;
    size_t rf_line_bytes   = sizeof(complex)*m_num_time_samples;

    // setup pre-computed convolution kernel and Hilbert transformer.
    m_device_excitation_fft = DeviceBufferRAII<complex>::u_ptr(new DeviceBufferRAII<complex>(rf_line_bytes));
    std::cout << "Number of excitation samples: " << m_excitation.samples.size() << std::endl;
    // convert to complex with zero imaginary part.
    std::vector<std::complex<float> > temp(m_num_time_samples);
    for (size_t i = 0; i < m_excitation.samples.size(); i++) {
        temp[i] = std::complex<float>(m_excitation.samples[i], 0.0f);
    }
    cudaErrorCheck( cudaMemcpy(m_device_excitation_fft->data(), temp.data(), rf_line_bytes, cudaMemcpyHostToDevice) );
    //dump_device_memory((std::complex<float>*)m_device_excitation_fft.data(), m_num_time_samples, "complex_exitation.txt");

    m_fft_plan = CufftPlanRAII::u_ptr(new CufftPlanRAII(m_num_time_samples, CUFFT_C2C, 1));

    // compute FFT of excitation signal and add the Hilbert transform
    cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), m_device_excitation_fft->data(), m_device_excitation_fft->data(), CUFFT_FORWARD) );
    auto mask = discrete_hilbert_mask<std::complex<float> >(m_num_time_samples);
    DeviceBufferRAII<complex> device_hilbert_mask(rf_line_bytes);
    cudaErrorCheck( cudaMemcpy(device_hilbert_mask.data(), mask.data(), rf_line_bytes, cudaMemcpyHostToDevice) );
    
    ScaleSignalKernel<<<m_num_time_samples/128, 128>>>(m_device_excitation_fft->data(), 1.0f/m_num_time_samples, m_num_time_samples);
    
    MultiplyFftKernel<<<m_num_time_samples/128, 128>>>(m_device_excitation_fft->data(), device_hilbert_mask.data(), m_num_time_samples);
    
    //dump_device_memory((std::complex<float>*) m_device_excitation_fft->data(), m_num_time_samples, "complex_excitation_fft.txt");
}


void CudaSplineAlgorithm2::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    m_scan_seq = new_scan_sequence;

    // HACK: Temporarily limited to the hardcoded value for m_num_time_samples
    auto num_rf_samples = compute_num_rf_samples(m_sim_params.sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);
    //std::cout << "num_rf_samples: " << num_rf_samples << std::endl;
    if (num_rf_samples > m_num_time_samples) {
        throw std::runtime_error("Too many RF samples required. TODO: remove limitation");
    }

    size_t num_beams = m_scan_seq->get_num_lines();
    // avoid reallocating memory if not necessary.
    if (m_num_beams_allocated < static_cast<int>(num_beams)) {
        std::cout << "Allocating HOST and DEVICE memory: had previously allocated memory for " << m_num_beams_allocated << " beams.\n";
    } else {
        return;
    }

    // allocate host and device memory related to RF lines
    size_t time_proj_bytes = sizeof(float)*m_num_time_samples;
    size_t rf_line_bytes   = sizeof(complex)*m_num_time_samples;
    m_device_time_proj.resize(m_num_cuda_streams);
    m_device_rf_lines.resize(m_num_cuda_streams);
    m_device_rf_lines_env.resize(m_num_cuda_streams);
    for (size_t i = 0; i < m_num_cuda_streams; i++) {
        m_device_time_proj[i]    = std::move(DeviceBufferRAII<float>::u_ptr   ( new DeviceBufferRAII<float>(time_proj_bytes)) ); 
        m_device_rf_lines[i]     = std::move(DeviceBufferRAII<complex>::u_ptr ( new DeviceBufferRAII<complex>(rf_line_bytes)) );
        m_device_rf_lines_env[i] = std::move(DeviceBufferRAII<float>::u_ptr   ( new DeviceBufferRAII<float>(time_proj_bytes)) ); 
    }

    // allocate host memory for all RF lines
    m_host_rf_lines.resize(num_beams);
    for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
        m_host_rf_lines[beam_no] = std::move(HostPinnedBufferRAII<float>::u_ptr( new HostPinnedBufferRAII<float>(time_proj_bytes)) );
    }

    m_num_beams_allocated = static_cast<int>(num_beams);
}

void CudaSplineAlgorithm2::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_num_scatterers = new_scatterers->num_scatterers();
        
    auto spline_scatterers = std::dynamic_pointer_cast<SplineScatterers>(new_scatterers);
    if (!spline_scatterers) {
        throw std::runtime_error("Cast failed");
    }
    copy_scatterers_to_device(spline_scatterers);
}


}   // end namespace