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
#include "cuda_debug_utils.h"
#include "device_launch_parameters.h"

// Slice a 3D lookup table through plane defined by two unit vectors.
// X- and y-components of grid determines the number of samples.
// NOTE: Number of threads in each block should be one.
__global__ void SliceLookupTable(float3 origin,
                                 float3 dir0,
                                 float3 dir1,
                                 float* output,
                                 cudaTextureObject_t lut_tex) {
    const int global_idx = blockIdx.x*gridDim.x + blockIdx.y;
    
    // FORMULA: offset = dim0*num_samples1 + dim1
    const int idx0 = blockIdx.x;  // idx0 = 0..gridDim.x
    const int idx1 = blockIdx.y;  // idx1 = 1..gridDim.y

    // Map to normalized distance in [0.0, 1.0]
    const auto normalized_dist0 = static_cast<float>(idx0)/(gridDim.x-1);
    const auto normalized_dist1 = static_cast<float>(idx1)/(gridDim.y-1);

    const auto tex_pos = origin + dir0*normalized_dist0 + dir1*normalized_dist1;
    output[global_idx] = tex3D<float>(lut_tex, tex_pos.x, tex_pos.y, tex_pos.z);
}

namespace bcsim {
GpuBaseAlgorithm::GpuBaseAlgorithm()
    : m_param_cuda_device_no(0),
      m_can_change_cuda_device(true),
      m_param_num_cuda_streams(2),
      m_num_time_samples(8192),  // TODO: remove this limitation
      m_num_beams_allocated(-1),
      m_param_threads_per_block(128),
      m_store_kernel_details(false),
      m_device_beam_profile(nullptr)
{
    // ensure that CUDA device properties is stored
    save_cuda_device_properties();
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
        save_cuda_device_properties();
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
    } else if (key == "store_kernel_details") {
        if ((value == "on") || (value == "true")) {
            m_store_kernel_details = true;
        } else if ((value == "off") || (value == "false")) {
            m_store_kernel_details = false;
        } else {
            throw std::runtime_error("invalid value");
        }
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

void GpuBaseAlgorithm::save_cuda_device_properties() {
    const auto num_devices = get_num_cuda_devices();
    if (m_param_cuda_device_no < 0 || m_param_cuda_device_no >= num_devices) {
        throw std::runtime_error("illegal CUDA device number");
    }
    cudaErrorCheck( cudaGetDeviceProperties(&m_cur_device_prop, m_param_cuda_device_no) );

    if (m_param_verbose) {
        const auto& p = m_cur_device_prop;
        std::cout << "=== CUDA Device " << m_param_cuda_device_no << ": " << p.name << std::endl;
        std::cout << "Compute capability: "         << p.major << "." << p.minor << std::endl;
        std::cout << "ECCEnabled: "                 << p.ECCEnabled                 << std::endl;
        std::cout << "asyncEngineCount: "           << p.asyncEngineCount           << std::endl;
        std::cout << "canMapHostMemory: "           << p.canMapHostMemory           << std::endl; 
        std::cout << "clockRate: "                  << p.clockRate                  << std::endl;
        std::cout << "computeMode: "                << p.computeMode                << std::endl;
        std::cout << "concurrentKernels: "          << p.concurrentKernels          << std::endl;
        std::cout << "integrated: "                 << p.integrated                 << std::endl;
        std::cout << "kernelExecTimeoutEnabled: "   << p.kernelExecTimeoutEnabled   << std::endl;
        std::cout << "l2CacheSize: "                << p.l2CacheSize                << std::endl;
        std::cout << "maxGridSize: [" << p.maxGridSize[0] << "," << p.maxGridSize[1] << "," << p.maxGridSize[2] << "]\n";
        std::cout << "maxThreadsPerBlock: "         << p.maxThreadsPerBlock         << std::endl;
        std::cout << "memoryBusWidth: "             << p.memoryBusWidth             << std::endl;
        std::cout << "multiProcessorCount: "        << p.multiProcessorCount        << std::endl;
        std::cout << "totalGlobMem: "               << p.totalGlobalMem             << std::endl;
    }
}

void GpuBaseAlgorithm::simulate_lines(std::vector<std::vector<std::complex<bc_float> > >&  /*out*/ rf_lines) {
    m_can_change_cuda_device = false;
    
    if (m_stream_wrappers.size() == 0) {
        create_cuda_stream_wrappers(m_param_num_cuda_streams);
    }

    if (m_store_kernel_details) {
        m_debug_data.clear();
    }
    
    auto num_lines = m_scan_seq->get_num_lines();
    if (num_lines < 1) {
        throw std::runtime_error("No scanlines in scansequence");
    }

    if (!m_beam_profile_configured) {
        throw std::runtime_error("No beam profile is configured");
    }
    
    // compute the number of blocks needed to project all scatterers and check that
    // it is not more than what is supported by the device.
    int num_blocks = round_up_div(m_num_scatterers, m_param_threads_per_block);
    if (num_blocks > m_cur_device_prop.maxGridSize[0]) {
        throw std::runtime_error("required number of x-blocks is larger than device supports");
    }

    // no delay compenasation is needed when returning the projections only
    size_t delay_compensation_num_samples = static_cast<size_t>(m_excitation.center_index);
    const auto num_return_samples = compute_num_rf_samples(m_param_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);
    
    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_param_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();

        std::unique_ptr<EventTimerRAII> event_timer;
        if (m_store_kernel_details) {
            event_timer = std::unique_ptr<EventTimerRAII>(new EventTimerRAII(cur_stream));
            m_debug_data["stream_numbers"].push_back(static_cast<double>(stream_no));
            event_timer->restart();
        }

        if (m_param_verbose) {
            std::cout << "beam_no = " << beam_no << ", stream_no = " << stream_no << std::endl;
        }

        auto scanline = m_scan_seq->get_scanline(beam_no);
        int threads_per_line = 128;
        auto rf_ptr = m_device_time_proj[stream_no]->data();

        // clear time projections (safer than cudaMemsetAsync)
        const auto complex_zero = make_cuComplex(0.0f, 0.0f);
        if (m_store_kernel_details) {
            event_timer->restart();
        }
        MemsetKernel<cuComplex><<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(rf_ptr,
                                                                                                          complex_zero,
                                                                                                          m_num_time_samples);
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_memset_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        projection_kernel(stream_no, scanline, num_blocks);
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_projection_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        // in-place forward FFT
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_FORWARD) );
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_forward_fft_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }
        
        // multiply with FFT of impulse response w/Hilbert transform
        MultiplyFftKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(rf_ptr,
                                                                                                    m_device_excitation_fft->data(),
                                                                                                    m_num_time_samples);
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_multiply_fft_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        // in-place inverse FFT
        cufftErrorCheck( cufftExecC2C(m_fft_plan->get(), rf_ptr, rf_ptr, CUFFT_INVERSE) );
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_inverse_fft_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        // IQ demodulation (+decimate?)
        const auto f_demod = m_excitation.demod_freq;
        const float norm_f_demod = f_demod/m_excitation.sampling_frequency;
        const float PI = static_cast<float>(4.0*std::atan(1));
        const auto normalized_angular_freq = 2*PI*norm_f_demod;
        DemodulateKernel<<<m_num_time_samples/threads_per_line, threads_per_line, 0, cur_stream>>>(rf_ptr, normalized_angular_freq, m_num_time_samples);
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_demodulate_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        // copy to host. Same memory layout?
        const auto num_bytes_iq = sizeof(std::complex<float>)*m_num_time_samples;
        cudaErrorCheck( cudaMemcpyAsync(m_host_rf_lines[beam_no]->data(), rf_ptr, num_bytes_iq, cudaMemcpyDeviceToHost, cur_stream) ); 
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_memcpy_ms"].push_back(elapsed_ms);
        }
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    // TODO: eliminate unneccessary data copying: it would e.g. be better to
    // only copy what is needed in the above kernel.
    rf_lines.clear();
    for (size_t line_no = 0; line_no < num_lines; line_no++) {
        std::vector<std::complex<bc_float>> temp_samples; // .reserve
        for (size_t i = 0; i < num_return_samples; i += m_radial_decimation) {
            temp_samples.push_back(m_host_rf_lines[line_no]->data()[i+delay_compensation_num_samples]);
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
        std::cout << "num_rf_samples = " << num_rf_samples << std::endl;
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

void GpuBaseAlgorithm::set_analytical_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting analytical beam profile for GPU algorithm" << std::endl;
    const auto analytical_profile = std::dynamic_pointer_cast<GaussianBeamProfile>(beam_profile);
    if (!analytical_profile) throw std::runtime_error("GpuBaseAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfile::ANALYTICAL;

    m_analytical_sigma_lat = analytical_profile->getSigmaLateral();
    m_analytical_sigma_ele = analytical_profile->getSigmaElevational();
}

void GpuBaseAlgorithm::set_lookup_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting LUT profile for GPU algorithm" << std::endl;
    const auto lut_beam_profile = std::dynamic_pointer_cast<LUTBeamProfile>(beam_profile);
    if (!lut_beam_profile) throw std::runtime_error("GpuBaseAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfile::LOOKUP;

    int num_samples_rad = lut_beam_profile->getNumSamplesRadial();
    int num_samples_lat = lut_beam_profile->getNumSamplesLateral();
    int num_samples_ele = lut_beam_profile->getNumSamplesElevational();
    std::cout << "=== set_beam_profile() ===" << std::endl;
    std::cout << "num_samples_rad: " << num_samples_rad << std::endl;
    std::cout << "num_samples_lat: " << num_samples_lat << std::endl;
    std::cout << "num_samples_ele: " << num_samples_ele << std::endl;
        
    const auto r_range = lut_beam_profile->getRangeRange();
    const auto l_range = lut_beam_profile->getLateralRange();
    const auto e_range = lut_beam_profile->getElevationalRange();

    // map to linear memory with correct 3D layout
    const auto total = num_samples_rad*num_samples_lat*num_samples_ele;
    std::vector<float> temp_samples;
    temp_samples.reserve(total);
    for (int zi = 0; zi < num_samples_rad; zi++) {
        for (int yi = 0; yi < num_samples_lat; yi++) {
            for (int xi = 0; xi < num_samples_ele; xi++) {
                const auto x = l_range.first + xi*(l_range.last-l_range.first)/(num_samples_lat-1);
                const auto y = e_range.first + yi*(e_range.last-e_range.first)/(num_samples_ele-1);
                const auto z = r_range.first + zi*(r_range.last-r_range.first)/(num_samples_rad-1);
                temp_samples.push_back(lut_beam_profile->sampleProfile(z, x, y));
            }
        }
    }
    m_device_beam_profile = DeviceBeamProfileRAII::u_ptr(new DeviceBeamProfileRAII(DeviceBeamProfileRAII::TableExtent3D(num_samples_lat, num_samples_ele, num_samples_rad),
                                                                                    temp_samples));
    // store spatial extent of profile.
    m_lut_r_min = r_range.first;
    m_lut_r_max = r_range.last;
    m_lut_l_min = l_range.first;
    m_lut_l_max = l_range.last;
    m_lut_e_min = e_range.first;
    m_lut_e_max = e_range.last;

    std::cout << "Created a new DeviceBeamProfileRAII.\n";
    
    // Slice the 3D texture and write as RAW file to disk.    
    const auto write_raw = [&](float3 origin, float3 dir0, float3 dir1, std::string raw_file) {
        const int num_samples = 1024;
        const int total_num_samples = num_samples*num_samples;
        const int num_bytes = sizeof(float)*total_num_samples;
        DeviceBufferRAII<float> device_slice(static_cast<size_t>(num_bytes));
            
        dim3 grid_size(num_samples, num_samples, 1);
        dim3 block_size(1, 1, 1);
        SliceLookupTable<<<grid_size, block_size>>>(origin, dir0, dir1,
                                                    device_slice.data(),
                                                    m_device_beam_profile->get());
        cudaErrorCheck( cudaDeviceSynchronize() );
        dump_device_buffer_as_raw_file(device_slice, raw_file);
    };

    const std::string raw_path("d:/temp/raw_lookup_table/");
    // slice in the middle lateral-elevational plane (radial dist is 0.5)
    write_raw(make_float3(0.0f, 0.0f, 0.5f),
                make_float3(1.0f, 0.0f, 0.0f),
                make_float3(0.0f, 1.0f, 0.0f),
                raw_path + "lut_slice_lat_ele.raw");
    // slice the middle lateral-radial plane (elevational dist is 0.5)
    write_raw(make_float3(0.0f, 0.5f, 0.0f),
                make_float3(1.0f, 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                raw_path + "lut_slice_lat_rad.raw");
    // slice the middle elevational-radial plane (lateral dist is 0.5)
    write_raw(make_float3(0.5f, 0.0f, 0.0f),
                make_float3(0.0f, 1.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                raw_path + "lut_slice_ele_rad.raw");
}

}   // end namespace

