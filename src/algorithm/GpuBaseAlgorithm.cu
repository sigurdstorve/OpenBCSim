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
#include <complex>
#include "GpuBaseAlgorithm.cuh"
#include "gpu_alg_common.cuh"
#include "common_utils.hpp" // for compute_num_rf_samples
#include "discrete_hilbert_mask.hpp"

namespace bcsim {
GpuBaseAlgorithm::GpuBaseAlgorithm()
    : m_param_cuda_device_no(0),
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
        m_param_cuda_device_no = device_no;
        cudaErrorCheck(cudaSetDevice(m_param_cuda_device_no));
        print_cuda_device_properties(m_param_cuda_device_no);
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
    } else if (key == "noise_amplitude") {
        throw std::runtime_error("noise is not yet implemented in GPU algorithms");
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

void GpuBaseAlgorithm::set_beam_profile(IBeamProfile::s_ptr beam_profile) {
    auto gaussian_profile = std::dynamic_pointer_cast<bcsim::GaussianBeamProfile>(beam_profile);
    if (!gaussian_profile) {
        throw std::runtime_error("GPU algorithm currently only supports analytical beam profiles");
    }
    m_beam_profile = gaussian_profile;   
}


void GpuBaseAlgorithm::simulate_lines(std::vector<std::vector<std::complex<bc_float> > >&  /*out*/ rf_lines) {
    m_can_change_cuda_device = false;
    
    if (m_stream_wrappers.size() == 0) {
        create_cuda_stream_wrappers(m_param_num_cuda_streams);
    }
    
    auto num_lines = m_scan_seq->get_num_lines();
    if (num_lines < 1) {
        throw std::runtime_error("No scanlines in scansequence");
    }

    if (m_beam_profile == nullptr) {
        throw std::runtime_error("No beam profile is set");
    }
    
    // no delay compenasation is needed when returning the projections only
    size_t delay_compensation_num_samples = static_cast<size_t>(m_excitation.center_index);
    const auto num_return_samples = compute_num_rf_samples(m_param_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);
    
    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_param_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();

        if (m_param_verbose) {
            std::cout << "beam_no = " << beam_no << ", stream_no = " << stream_no << std::endl;
        }

        auto scanline = m_scan_seq->get_scanline(beam_no);
        int threads_per_line = 128;
        auto rf_ptr = m_device_time_proj[stream_no]->data();

        // clear time projections (safer than cudaMemsetAsync)
        const auto complex_zero = make_cuComplex(0.0f, 0.0f);
        MemsetKernel<cuComplex><<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(rf_ptr,
                                                                                                          complex_zero,
                                                                                                          m_num_time_samples);


        projection_kernel(stream_no, scanline);

        // in-place forward FFT            
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_FORWARD) );

        // multiply with FFT of impulse response (can include Hilbert transform also)
        MultiplyFftKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(rf_ptr,
                                                                                                    m_device_excitation_fft->data(),
                                                                                                    m_num_time_samples);

        // in-place inverse FFT
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_INVERSE) );
        
        // copy to host. Same memory layout?
        const auto num_bytes_iq = sizeof(std::complex<float>)*m_num_time_samples;
        cudaErrorCheck( cudaMemcpyAsync(m_host_rf_lines[beam_no]->data(), rf_ptr, num_bytes_iq, cudaMemcpyDeviceToHost, cur_stream) ); 
            
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    // TODO: eliminate unneccessary data copying: it would e.g. be better to
    // only copy what is needed in the above kernel.
    // TODO: Decimate signal.
    rf_lines.clear();
    std::vector<std::complex<bc_float>> temp_samples(num_return_samples);
    for (size_t line_no = 0; line_no < num_lines; line_no++) {
        for (size_t i = 0; i < num_return_samples; i++) {
            temp_samples[i] = m_host_rf_lines[line_no]->data()[i+delay_compensation_num_samples];
        }
        rf_lines.push_back(temp_samples);
    }
}

void GpuBaseAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
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
    MultiplyFftKernel<<<m_num_time_samples/128, 128>>>(m_device_excitation_fft->data(), device_hilbert_mask.data(), m_num_time_samples);
    //dump_device_memory((std::complex<float>*) m_device_excitation_fft->data(), m_num_time_samples, "complex_excitation_fft.txt");
}


void GpuBaseAlgorithm::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    m_can_change_cuda_device = false;
    
    m_scan_seq = new_scan_sequence;

    // HACK: Temporarily limited to the hardcoded value for m_num_time_samples
    auto num_rf_samples = compute_num_rf_samples(m_param_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);
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
    const auto device_iq_line_bytes = sizeof(complex)*m_num_time_samples;
    const auto host_iq_line_bytes   = sizeof(std::complex<float>)*m_num_time_samples;

    m_device_time_proj.resize(m_param_num_cuda_streams);
    for (size_t i = 0; i < m_param_num_cuda_streams; i++) {
        m_device_time_proj[i]    = std::move(DeviceBufferRAII<complex>::u_ptr ( new DeviceBufferRAII<complex>(device_iq_line_bytes)) ); 
    }

    // allocate host memory for all RF lines
    m_host_rf_lines.resize(num_beams);
    for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
        m_host_rf_lines[beam_no] = std::move(HostPinnedBufferRAII<std::complex<float>>::u_ptr( new HostPinnedBufferRAII<std::complex<float>>(host_iq_line_bytes)) );
    }

    m_num_beams_allocated = static_cast<int>(num_beams);
}

}   // end namespace

