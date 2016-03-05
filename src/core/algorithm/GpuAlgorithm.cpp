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
#include <cuda.h>
#include <stdexcept>
#include <iostream>
#include <complex>
#include <tuple> // for std::tie
#include "GpuAlgorithm.hpp"
#include "common_utils.hpp" // for compute_num_rf_samples
#include "../discrete_hilbert_mask.hpp"
#include "cuda_debug_utils.h"
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "cuda_kernels_c_interface.h"
#include "common_definitions.h" // for MAX_NUM_CUDA_STREAMS and MAX_SPLINE_DEGREE
#include "../bspline.hpp"

namespace bcsim {
GpuAlgorithm::GpuAlgorithm()
    : m_param_cuda_device_no(0),
      m_can_change_cuda_device(true),
      m_param_num_cuda_streams(2), // TODO: What if this value is bigger than max num streams...
      m_num_time_samples(8192),  // TODO: remove this limitation
      m_num_beams_allocated(-1),
      m_param_threads_per_block(128),
      m_store_kernel_details(false),
      m_device_random_buffer(nullptr)
{
    // ensure that CUDA device properties is stored
    save_cuda_device_properties();

    create_dummy_lut_profile();
}

int GpuAlgorithm::get_num_cuda_devices() const {
    int device_count;
    cudaErrorCheck( cudaGetDeviceCount(&device_count) );
    return device_count;
}

void GpuAlgorithm::set_parameter(const std::string& key, const std::string& value) {
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
        if (num_streams > MAX_NUM_CUDA_STREAMS) {
            throw std::runtime_error("number of CUDA streams exceeds MAX_NUM_CUDA_STREAMS");
        }
        if (num_streams <= 0) {
            throw std::runtime_error("number of CUDA streams must be more than zero");
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

void GpuAlgorithm::create_cuda_stream_wrappers(int num_streams) {
    m_stream_wrappers.clear();
    for (int i = 0; i < num_streams; i++) {
        m_stream_wrappers.push_back(std::move(CudaStreamRAII::u_ptr(new CudaStreamRAII)));
    }
    m_can_change_cuda_device = false;
}

void GpuAlgorithm::save_cuda_device_properties() {
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

void GpuAlgorithm::simulate_lines(std::vector<std::vector<std::complex<float> > >&  /*out*/ rf_lines) {
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

    if (m_cur_beam_profile_type == BeamProfileType::NOT_CONFIGURED) {
        throw std::runtime_error("No beam profile is configured");
    }

    // TODO: If all beams have the same timestamp, first render to fixed scatterers
    // in device memory and then simulate with the fixed algorithm
    bool use_optimized_spline_kernel = false;
    if (m_scan_seq->all_timestamps_equal && (m_device_spline_datasets.get_num_datasets() > 0)) {
        const auto timestamp = m_scan_seq->get_scanline(0).get_timestamp();
        m_device_rendered_spline_datasets.render(m_device_spline_datasets, timestamp);
        use_optimized_spline_kernel = true;
        cudaErrorCheck(cudaDeviceSynchronize());
    }

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
        auto rf_ptr = m_device_time_proj->data() + beam_no*m_num_time_samples;

        // clear time projections (safer than cudaMemsetAsync)
        const auto complex_zero = make_cuComplex(0.0f, 0.0f);
        if (m_store_kernel_details) {
            event_timer->restart();
        }
        launch_MemsetKernel<cuComplex>(m_num_time_samples/threads_per_line, threads_per_line, cur_stream, rf_ptr, complex_zero, m_num_time_samples);

        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_memset_ms"].push_back(elapsed_ms);
            event_timer->restart();
        }

        // project fixed scatterers
        for (size_t dset_idx = 0; dset_idx < m_device_fixed_datasets.get_num_datasets(); dset_idx++) {
            const auto device_dataset = m_device_fixed_datasets.get_dataset(dset_idx);
            const auto num_scatterers = device_dataset->get_num_scatterers();
            const int num_blocks = round_up_div(num_scatterers, m_param_threads_per_block);
            if (num_blocks > m_cur_device_prop.maxGridSize[0]) {
                throw std::runtime_error("required number of x-blocks is larger than device supports (fixed scatterers)");
            }
            fixed_projection_kernel(stream_no, scanline, num_blocks, rf_ptr, device_dataset);

            if (m_store_kernel_details) {
                const auto elapsed_ms = static_cast<double>(event_timer->stop());
                m_debug_data["fixed_projection_kernel_ms"].push_back(elapsed_ms);
                event_timer->restart();
            }
        }

        // project spline scatterers
        if (use_optimized_spline_kernel) {
            for (size_t dset_idx = 0; dset_idx < m_device_rendered_spline_datasets.get_num_datasets(); dset_idx++) {
                const auto device_dataset = m_device_rendered_spline_datasets.get_dataset(dset_idx);
                const auto num_scatterers = device_dataset->get_num_scatterers();
                const int num_blocks = round_up_div(num_scatterers, m_param_threads_per_block);
                if (num_blocks > m_cur_device_prop.maxGridSize[0]) {
                    throw std::runtime_error("required number of x-blocks is larger than device supports (spline scatterers)");
                }
                fixed_projection_kernel(stream_no, scanline, num_blocks, rf_ptr, device_dataset);
            }
        } else {
            for (size_t dset_idx = 0; dset_idx < m_device_spline_datasets.get_num_datasets(); dset_idx++) {
                const auto device_dataset = m_device_spline_datasets.get_dataset(dset_idx);
                const auto num_scatterers = device_dataset->get_num_scatterers();
                const int num_blocks = round_up_div(num_scatterers, m_param_threads_per_block);
                if (num_blocks > m_cur_device_prop.maxGridSize[0]) {
                    throw std::runtime_error("required number of x-blocks is larger than device supports (spline scatterers)");
                }
                spline_projection_kernel(stream_no, scanline, num_blocks, rf_ptr, device_dataset);
    
                if (m_store_kernel_details) {
                    const auto elapsed_ms = static_cast<double>(event_timer->stop());
                    m_debug_data["spline_projection_kernel_ms"].push_back(elapsed_ms);
                }
            }
        }
    }

    // block to ensure that all operations are completed
    cudaErrorCheck( cudaDeviceSynchronize() );

    std::unique_ptr<EventTimerRAII> event_timer;
    if (m_store_kernel_details) {
        event_timer = std::unique_ptr<EventTimerRAII>(new EventTimerRAII(0));
        event_timer->restart();
    }

    // in-place batched forward FFT, using default stream 0
    cufftErrorCheck(cufftExecC2C(m_fft_plan->get(), m_device_time_proj->data(), m_device_time_proj->data(), CUFFT_FORWARD));
    if (m_store_kernel_details) {
        const auto elapsed_ms = static_cast<double>(event_timer->stop());
        m_debug_data["kernel_forward_fft_ms"].push_back(elapsed_ms);
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    // Multiply kernel
    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_param_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();
        
        std::unique_ptr<EventTimerRAII> event_timer;
        if (m_store_kernel_details) {
            event_timer = std::unique_ptr<EventTimerRAII>(new EventTimerRAII(cur_stream));
            event_timer->restart();
        }

        auto rf_ptr = m_device_time_proj->data() + beam_no*m_num_time_samples;

        // multiply with FFT of impulse response w/Hilbert transform
        int threads_per_line = 128;
        launch_MultiplyFftKernel(m_num_time_samples/threads_per_line, threads_per_line, cur_stream, rf_ptr, m_device_excitation_fft->data(), m_num_time_samples);
        if (m_store_kernel_details) {
            const auto elapsed_ms = static_cast<double>(event_timer->stop());
            m_debug_data["kernel_multiply_fft_ms"].push_back(elapsed_ms);
        }
    }

    // in-place batched backward FFT, using default stream 0
    if (m_store_kernel_details) {
        event_timer->restart();
    }
    cudaErrorCheck( cudaDeviceSynchronize() );
    cufftErrorCheck(cufftExecC2C(m_fft_plan->get(), m_device_time_proj->data(), m_device_time_proj->data(), CUFFT_INVERSE));
    if (m_store_kernel_details) {
        const auto elapsed_ms = static_cast<double>(event_timer->stop());
        m_debug_data["kernel_inverse_fft_ms"].push_back(elapsed_ms);
    }
    cudaErrorCheck( cudaDeviceSynchronize() );

    for (int beam_no = 0; beam_no < num_lines; beam_no++) {
        size_t stream_no = beam_no % m_param_num_cuda_streams;
        auto cur_stream = m_stream_wrappers[stream_no]->get();

        // compute current offset into device buffer
        auto rf_ptr = m_device_time_proj->data() + beam_no*m_num_time_samples;
        
        std::unique_ptr<EventTimerRAII> event_timer;
        if (m_store_kernel_details) {
            event_timer = std::unique_ptr<EventTimerRAII>(new EventTimerRAII(cur_stream));
            event_timer->restart();
        }

        // IQ demodulation (+decimate?)
        int threads_per_line = 128;
        const auto f_demod = m_excitation.demod_freq;
        const float norm_f_demod = f_demod/m_excitation.sampling_frequency;
        const float PI = static_cast<float>(4.0*std::atan(1));
        const auto normalized_angular_freq = 2*PI*norm_f_demod;
        launch_DemodulateKernel(m_num_time_samples/threads_per_line, threads_per_line, cur_stream, rf_ptr, normalized_angular_freq, m_num_time_samples);
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

    // no delay compenasation is needed when returning the projections only
    size_t delay_compensation_num_samples = static_cast<size_t>(m_excitation.center_index);
    const auto num_return_samples = compute_num_rf_samples(m_param_sound_speed, m_scan_seq->line_length, m_excitation.sampling_frequency);

    // TODO: eliminate unneccessary data copying: it would e.g. be better to
    // only copy what is needed in the above kernel.
    rf_lines.clear();
    for (size_t line_no = 0; line_no < num_lines; line_no++) {
        std::vector<std::complex<float>> temp_samples; // .reserve
        for (size_t i = 0; i < num_return_samples; i += m_radial_decimation) {
            temp_samples.push_back(m_host_rf_lines[line_no]->data()[i+delay_compensation_num_samples]);
        }
        rf_lines.push_back(temp_samples);
    }
}

void GpuAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
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

    // create a temporary plan for computing forward FFT of excitation
    auto excitation_fft_plan = CufftPlanRAII::u_ptr(new CufftPlanRAII(m_num_time_samples, CUFFT_C2C, 1));

    // compute FFT of excitation signal and add the Hilbert transform
    cufftErrorCheck( cufftExecC2C(excitation_fft_plan->get(), m_device_excitation_fft->data(), m_device_excitation_fft->data(), CUFFT_FORWARD) );
    auto mask = discrete_hilbert_mask<std::complex<float> >(m_num_time_samples);
    DeviceBufferRAII<complex> device_hilbert_mask(rf_line_bytes);
    cudaErrorCheck( cudaMemcpy(device_hilbert_mask.data(), mask.data(), rf_line_bytes, cudaMemcpyHostToDevice) );
    
    cudaStream_t cuda_stream = 0;
    launch_ScaleSignalKernel(m_num_time_samples/128, 128, cuda_stream, m_device_excitation_fft->data(), 1.0f/m_num_time_samples, m_num_time_samples);
    launch_MultiplyFftKernel(m_num_time_samples/128, 128, cuda_stream, m_device_excitation_fft->data(), device_hilbert_mask.data(), m_num_time_samples);
}


void GpuAlgorithm::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
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
    if (m_num_beams_allocated != num_beams) {
        std::cout << "Reconfiguring cuFFT batched plan\n";
        std::cout << "m_num_time_samples is " << m_num_time_samples << std::endl;
        const auto num_samples = static_cast<int>(m_num_time_samples);
        const auto batch = static_cast<int>(num_beams);
        const int rank = 1;
        int dims[] = {m_num_time_samples};
        m_fft_plan = CufftBatchedPlanRAII::u_ptr(new CufftBatchedPlanRAII(rank, dims, num_samples, CUFFT_C2C, batch));
        std::cout << "batch = " << batch << std::endl;

        // allocate host and device memory related to RF lines
        const auto device_iq_line_bytes = sizeof(complex)*m_num_time_samples;
        const auto host_iq_line_bytes   = sizeof(std::complex<float>)*m_num_time_samples;

        std::cout << "Reallocating HOST and DEVICE memory\n";
        m_device_time_proj = DeviceBufferRAII<complex>::u_ptr ( new DeviceBufferRAII<complex>(device_iq_line_bytes*num_beams));

        // allocate host memory for all RF lines
        m_host_rf_lines.resize(num_beams);
        for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
            m_host_rf_lines[beam_no] = std::move(HostPinnedBufferRAII<std::complex<float>>::u_ptr( new HostPinnedBufferRAII<std::complex<float>>(host_iq_line_bytes)) );
        }

        m_num_beams_allocated = static_cast<int>(num_beams);
    }
}

void GpuAlgorithm::set_analytical_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting analytical beam profile for GPU algorithm" << std::endl;
    const auto analytical_profile = std::dynamic_pointer_cast<GaussianBeamProfile>(beam_profile);
    if (!analytical_profile) throw std::runtime_error("GpuAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfileType::ANALYTICAL;

    m_analytical_sigma_lat = analytical_profile->getSigmaLateral();
    m_analytical_sigma_ele = analytical_profile->getSigmaElevational();
}

void GpuAlgorithm::set_lookup_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting LUT profile for GPU algorithm" << std::endl;
    const auto lut_beam_profile = std::dynamic_pointer_cast<LUTBeamProfile>(beam_profile);
    if (!lut_beam_profile) throw std::runtime_error("GpuAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfileType::LOOKUP;

    int num_samples_rad = lut_beam_profile->getNumSamplesRadial();
    int num_samples_lat = lut_beam_profile->getNumSamplesLateral();
    int num_samples_ele = lut_beam_profile->getNumSamplesElevational();
    std::cout << "=== set_lookup_profile() ===" << std::endl;
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
    
    if (false) {
        const std::string raw_lut_path("d:/temp/raw_lookup_table/");
        dump_orthogonal_lut_slices(raw_lut_path);
        // write extents
        std::ofstream out_stream;
        out_stream.open(raw_lut_path + "/extents.txt");
        out_stream << m_lut_r_min << " " << m_lut_r_max << std::endl;
        out_stream << m_lut_l_min << " " << m_lut_l_max << std::endl;
        out_stream << m_lut_e_min << " " << m_lut_e_max << std::endl;
    }

}

void GpuAlgorithm::dump_orthogonal_lut_slices(const std::string& raw_path) {
    const auto write_raw = [&](float3 origin, float3 dir0, float3 dir1, std::string raw_file) {
        const int num_samples = 1024;
        const int total_num_samples = num_samples*num_samples;
        const int num_bytes = sizeof(float)*total_num_samples;
        DeviceBufferRAII<float> device_slice(static_cast<size_t>(num_bytes));
            
        //dim3 grid_size(num_samples, num_samples, 1);
        //dim3 block_size(1, 1, 1);
        const cudaStream_t cuda_stream = 0;
        launch_SliceLookupTable(num_samples, num_samples, 1, cuda_stream,
                                origin, dir0, dir1, device_slice.data(), m_device_beam_profile->get());
        cudaErrorCheck( cudaDeviceSynchronize() );
        dump_device_buffer_as_raw_file(device_slice, raw_file);
    };

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

    // 6 equally spaced lateral-elevational slices of [0.0, 1.0]
    for (int i = 0; i <=5; i++) {
        write_raw(make_float3(0.0f, 0.0f, static_cast<float>(i)/5),
                  make_float3(1.0f, 0.0f, 0.0f),
                  make_float3(0.0f, 1.0f, 0.0f),
                  raw_path + "lut_slice_lat_ele_"+std::to_string(i)+".raw");
    }

}

void GpuAlgorithm::create_dummy_lut_profile() {
    const size_t n = 16;
    std::vector<float> dummy_samples(n*n*n, 0.0f);
    m_device_beam_profile = DeviceBeamProfileRAII::u_ptr(new DeviceBeamProfileRAII(DeviceBeamProfileRAII::TableExtent3D(n, n, n), dummy_samples));
}

void GpuAlgorithm::clear_fixed_scatterers() {
    m_device_fixed_datasets.clear();
}

void GpuAlgorithm::add_fixed_scatterers(FixedScatterers::s_ptr fixed_scatterers) {
    m_device_fixed_datasets.add(fixed_scatterers);
    m_can_change_cuda_device = false;
}

void GpuAlgorithm::clear_spline_scatterers() {
    m_device_spline_datasets.clear();
}

void GpuAlgorithm::add_spline_scatterers(SplineScatterers::s_ptr spline_scatterers) {
    m_can_change_cuda_device = false;
    m_device_spline_datasets.add(spline_scatterers);
}

void GpuAlgorithm::fixed_projection_kernel(int stream_no, const Scanline& scanline, int num_blocks, cuComplex* res_buffer, DeviceFixedScatterers::s_ptr dataset) {
    auto cur_stream = m_stream_wrappers[stream_no]->get();

    //dim3 grid_size(num_blocks, 1, 1);
    //dim3 block_size(m_param_threads_per_block, 1, 1);

    // prepare struct with parameters
    FixedAlgKernelParams params;
    params.point_xs          = dataset->get_xs_ptr();
    params.point_ys          = dataset->get_ys_ptr();
    params.point_zs          = dataset->get_zs_ptr();
    params.point_as          = dataset->get_as_ptr();
    params.rad_dir           = to_float3(scanline.get_direction());
    params.lat_dir           = to_float3(scanline.get_lateral_dir());
    params.ele_dir           = to_float3(scanline.get_elevational_dir());
    params.origin            = to_float3(scanline.get_origin());
    params.fs_hertz          = m_excitation.sampling_frequency;
    params.num_time_samples  = m_num_time_samples;
    params.sigma_lateral     = m_analytical_sigma_lat;
    params.sigma_elevational = m_analytical_sigma_ele;
    params.sound_speed       = m_param_sound_speed;
    params.res               = res_buffer;
    params.demod_freq        = m_excitation.demod_freq;
    params.num_scatterers    = dataset->get_num_scatterers(),
    params.lut_tex           = m_device_beam_profile->get();
    params.lut.r_min         = m_lut_r_min;
    params.lut.r_max         = m_lut_r_max;
    params.lut.l_min         = m_lut_l_min;
    params.lut.l_max         = m_lut_l_max;
    params.lut.e_min         = m_lut_e_min;
    params.lut.e_max         = m_lut_e_max;

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

    if (!m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<false, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<false, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<false, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (!m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<false, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<true, false, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && !m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<true, false, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && !use_lut) {
        launch_FixedAlgKernel<true, true, false>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else if (m_param_use_arc_projection && m_enable_phase_delay && use_lut) {
        launch_FixedAlgKernel<true, true, true>(num_blocks, m_param_threads_per_block, cur_stream, params);
    } else {
        throw std::logic_error("this should never happen");
    }
}

void GpuAlgorithm::spline_projection_kernel(int stream_no, const Scanline& scanline, int num_blocks, cuComplex* res_buffer, DeviceSplineScatterers::s_ptr dataset) {
    auto cur_stream = m_stream_wrappers[stream_no]->get();
    const auto cur_knots = dataset->get_knots();
    const auto num_cs    = dataset->get_num_cs();
    const auto spline_degree = dataset->get_spline_degree();

    // evaluate the basis functions and upload to constant memory.
    const auto num_nonzero = spline_degree+1;
    size_t eval_basis_offset_elements = num_nonzero*stream_no;
    std::vector<float> host_basis_functions(num_cs);
    for (int i = 0; i < num_cs; i++) {
        host_basis_functions[i] = bspline_storve::bsplineBasis(i, spline_degree, scanline.get_timestamp(), cur_knots);
    }

    //dim3 grid_size(num_blocks, 1, 1);
    //dim3 block_size(m_param_threads_per_block, 1, 1);

    // compute sum limits (inclusive)
    int cs_idx_start, cs_idx_end;
    std::tie(cs_idx_start, cs_idx_end) = bspline_storve::get_lower_upper_inds(cur_knots,
                                                                              scanline.get_timestamp(),
                                                                              spline_degree);
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
    params.control_xs                 = dataset->get_xs_ptr();
    params.control_ys                 = dataset->get_ys_ptr();
    params.control_zs                 = dataset->get_zs_ptr();
    params.control_as                 = dataset->get_as_ptr();
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
    params.NUM_SPLINES                = dataset->get_num_scatterers(),
    params.res                        = res_buffer;
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
        throw std::logic_error("spline_projection_kernel(): unknown beam profile type");
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

size_t GpuAlgorithm::get_total_num_scatterers() const {
    const auto total_num_fixed  = m_device_fixed_datasets.get_total_num_scatterers();
    const auto total_num_spline = m_device_spline_datasets.get_total_num_scatterers();
    return total_num_fixed+total_num_spline;
}

std::string GpuAlgorithm::get_parameter(const std::string& key) const {
    if (key == "num_cuda_devices") {
        int num_devices;
        cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
        return std::to_string(num_devices);
    } else if (key == "cur_device_name") {
        cudaDeviceProp prop;
        cudaErrorCheck( cudaGetDeviceProperties(&prop, m_param_cuda_device_no) );
        return prop.name;
    } else {
        return BaseAlgorithm::get_parameter(key);
    }
}

}   // end namespace

#endif // BCSIM_ENABLE_CUDA
