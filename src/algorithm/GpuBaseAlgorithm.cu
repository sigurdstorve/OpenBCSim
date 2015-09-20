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
#include <stdexcept>
#include <iostream>
#include "GpuBaseAlgorithm.cuh"
#include "gpu_alg_common.cuh"
#include "common_utils.hpp" // for compute_num_rf_samples

namespace bcsim {
GpuBaseAlgorithm::GpuBaseAlgorithm()
    : m_sound_speed(1540.0f),
      m_cuda_device_no(0),
      m_can_change_cuda_device(true),
      m_param_num_cuda_streams(2),
      m_num_time_samples(32768),  // TODO: remove this limitation
      m_beam_profile(nullptr),
      m_num_beams_allocated(-1),
      m_param_threads_per_block(128)
{
}

int GpuBaseAlgorithm::get_num_cuda_devices() const {
    int device_count;
    cudaErrorCheck( cudaGetDeviceCount(&device_count) );
    return device_count;
}

void GpuBaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "gpu_device") {
        if (!m_can_change_cuda_device) {
            throw std::runtime_error("cannot change CUDA device now");            
        }
        const auto device_count = get_num_cuda_devices();
        const int device_no = std::stoi(value);
        if (device_no < 0 || device_no >= device_count) {
            throw std::runtime_error("illegal device number");
        }
        m_cuda_device_no = device_no;
        cudaErrorCheck(cudaSetDevice(m_cuda_device_no));
        print_cuda_device_properties(m_cuda_device_no);
    } else if (key == "sound_speed") {
        m_sound_speed = std::stof(value);
    } else if (key == "cuda_streams") {
        const auto num_streams = std::stoi(value);
        if (num_streams <= 0) {
            throw std::runtime_error("invalid number of CUDA streams");
        }
        m_param_num_cuda_streams = num_streams;
    } else if (key == "threads_per_block") {
        const auto threads_per_block = std::stoi(value);
        if (threads_per_block <= 0) {
            throw std::runtime_error("invalid number of threads per block");            
        }
        m_param_threads_per_block = threads_per_block;
    } else {
        BaseAlgorithm::set_parameter(key, value);
    }
}

void GpuBaseAlgorithm::create_cuda_stream_wrappers(int num_streams) {
    m_stream_wrappers.clear();
    for (int i = 0; i < num_streams; i++) {
        m_stream_wrappers.push_back(std::move(CudaStreamRAII::u_ptr(new CudaStreamRAII)));
    }
    m_can_change_cuda_device = false;
}

void GpuBaseAlgorithm::print_cuda_device_properties(int device_no) const {
    const auto num_devices = get_num_cuda_devices();
    if (device_no < 0 || device_no >= num_devices) {
        throw std::runtime_error("illegal CUDA device number");
    }
    cudaDeviceProp prop;
    cudaErrorCheck( cudaGetDeviceProperties(&prop, device_no) );
    std::cout << "\n\n=== Device " << device_no << ": " << prop.name               << std::endl;
    std::cout << "totalGlobMem: "               << prop.totalGlobalMem             << std::endl;
    std::cout << "clockRate: "                  << prop.clockRate                  << std::endl;
    std::cout << "Compute capability: "         << prop.major << "." << prop.minor << std::endl;
    std::cout << "asyncEngineCount: "           << prop.asyncEngineCount           << std::endl;
    std::cout << "multiProcessorCount: "        << prop.multiProcessorCount        << std::endl;
    std::cout << "kernelExecTimeoutEnabled: "   << prop.kernelExecTimeoutEnabled   << std::endl;
    std::cout << "computeMode: "                << prop.computeMode                << std::endl;
    std::cout << "concurrentKernels: "          << prop.concurrentKernels          << std::endl;
    std::cout << "ECCEnabled: "                 << prop.ECCEnabled                 << std::endl;
    std::cout << "memoryBusWidth: "             << prop.memoryBusWidth             << std::endl;
}

void GpuBaseAlgorithm::simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines) {
    m_can_change_cuda_device = false;
    
    if (m_stream_wrappers.size() == 0) {
        create_cuda_stream_wrappers(m_param_num_cuda_streams);
    }
    
    auto num_lines      = m_scan_seq->get_num_lines();

    if (num_lines < 1) {
        throw std::runtime_error("No scanlines in scansequence");
    }

    if (m_beam_profile == nullptr) {
        throw std::runtime_error("No beam profile is set");
    }
    
    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_param_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();

        if (m_param_verbose) {
            std::cout << "beam_no = " << beam_no << ", stream_no = " << stream_no << std::endl;
        }

        auto scanline = m_scan_seq->get_scanline(beam_no);

        //std::cout << "origin: " << origin.x << " " << origin.y << " " << origin.z << std::endl;

        int threads_per_line = 128;
        // clear the time projection buffer the proper way (probably slightly slower than cudaMamSetAsync...)
        MemsetFloatKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_time_proj[stream_no]->data(),
                                                                                                    0.0f,
                                                                                                    m_num_time_samples);

        //if (beam_no==0) { dump_device_memory<float>(device_time_proj[stream_no]->data(), m_num_time_samples, "01_zeroed_rf_line_dump.txt"); }
        projection_kernel(stream_no, scanline);
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

        if (m_param_output_type == OutputType::ENVELOPE_DATA) {
            AbsComplexKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                       m_device_rf_lines_env[stream_no]->data(),
                                                                                                       m_num_time_samples);
        } else if (m_param_output_type == OutputType::RF_DATA) {
            RealPartKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(m_device_rf_lines[stream_no]->data(),
                                                                                                     m_device_rf_lines_env[stream_no]->data(),
                                                                                                     m_num_time_samples);
        } else if (m_param_output_type == OutputType::PROJECTIONS) {
            throw std::runtime_error("Output type PROJECTIONS is not yet supported");        
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


}   // end namespace

