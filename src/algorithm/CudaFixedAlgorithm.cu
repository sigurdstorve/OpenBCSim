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
#include "CudaFixedAlgorithm.cuh"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "device_launch_parameters.h" // for removing annoying MSVC intellisense error messages
#include "discrete_hilbert_mask.hpp"
#include "common_utils.hpp" // for compute_num_rf_samples
#include "gpu_alg_common.cuh" // for misc. CUDA kernels

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


CudaFixedAlgorithm::CudaFixedAlgorithm()
    : m_verbose(false),
      m_num_cuda_streams(2),
      m_num_time_samples(32768),  // TODO: remove this limitation
      m_num_beams_allocated(-1),
      m_beam_profile(nullptr),
      m_output_type("env")
{
}

void CudaFixedAlgorithm::simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines) {
    cudaErrorCheck(cudaSetDevice(m_cuda_device_no));
    m_can_change_cuda_device = false;
    
    if (m_stream_wrappers.size() == 0) {
        create_cuda_stream_wrappers(m_num_cuda_streams);
    }
    
    auto num_lines      = m_scan_seq->get_num_lines();

    if (num_lines < 1) {
        throw std::runtime_error("No scanlines in scansequence");
    }

    if (m_beam_profile == nullptr) {
        throw std::runtime_error("No beam profile is set");
    }
    
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

        if (m_output_type == "env") {
            // envelope detection
            AbsComplexKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                    m_device_rf_lines_env[stream_no]->data(),
                                                                                                    m_num_time_samples);
        } else if (m_output_type == "rf") {
            // rf data
            RealPartKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                     m_device_rf_lines_env[stream_no]->data(),
                                                                                                     m_num_time_samples);
            
        } else {
            throw std::logic_error("illegal output type");        
        }
        //if (beam_no==0) { dump_device_memory<float>(device_rf_lines_env[stream_no]->data(), m_num_time_samples, "05_rf_envelope.txt"); }
            
        // copy to host
        cudaErrorCheck( cudaMemcpyAsync(m_host_rf_lines[beam_no]->data(), m_device_rf_lines_env[stream_no]->data(), sizeof(float)*m_num_time_samples, cudaMemcpyDeviceToHost, cur_stream) ); 
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    // TODO: eliminate unneccessary data copying: it would e.g. be better to
    // only copy what is needed in the above kernel.
    
    const auto num_return_samples = compute_num_rf_samples(m_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);

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

void CudaFixedAlgorithm::copy_scatterers_to_device(FixedScatterers::s_ptr scatterers) {
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

void CudaFixedAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
    m_can_change_cuda_device = false;
    
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
    
    if (m_output_type == "env") {
        MultiplyFftKernel<<<m_num_time_samples/128, 128>>>(m_device_excitation_fft->data(), device_hilbert_mask.data(), m_num_time_samples);
    }
    //dump_device_memory((std::complex<float>*) m_device_excitation_fft->data(), m_num_time_samples, "complex_excitation_fft.txt");
}


void CudaFixedAlgorithm::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    m_can_change_cuda_device = false;
    m_scan_seq = new_scan_sequence;

    // HACK: Temporarily limited to the hardcoded value for m_num_time_samples
    auto num_rf_samples = compute_num_rf_samples(m_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);
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

void CudaFixedAlgorithm::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_can_change_cuda_device = false;
    m_num_scatterers = new_scatterers->num_scatterers();
        
    auto fixed_scatterers = std::dynamic_pointer_cast<FixedScatterers>(new_scatterers);
    if (!fixed_scatterers) {
        throw std::runtime_error("Cast failed");
    }
    copy_scatterers_to_device(fixed_scatterers);
}


}   // end namespace