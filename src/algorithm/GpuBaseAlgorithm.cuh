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
#pragma once
#include <vector>
#include <memory>
#include <cuda.h>
#include <cufft.h>
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "BaseAlgorithm.hpp"

namespace bcsim {

class GpuBaseAlgorithm : public BaseAlgorithm {
public:
    GpuBaseAlgorithm();
    
    virtual void set_parameter(const std::string& key, const std::string& value)        override;

    virtual void simulate_lines(std::vector<std::vector<std::complex<bc_float>> >&  /*out*/ rf_lines) override;
    
    // NOTE: currently requires that set_excitation is called first!
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)               override;

    virtual void set_excitation(const ExcitationSignal& new_excitation)                 override;
    
    virtual void set_analytical_profile(IBeamProfile::s_ptr beam_profile) override;

    virtual void set_lookup_profile(IBeamProfile::s_ptr beam_profile) override;

protected:
    // Debug functionality: slice the 3D texture and write as RAW file to disk.    
    void GpuBaseAlgorithm::dump_orthogonal_lut_slices(const std::string& raw_path);

    void create_cuda_stream_wrappers(int num_streams);
    
    int get_num_cuda_devices() const;
    
    void save_cuda_device_properties();
    
    // must be implemented in subclass
    virtual void projection_kernel(int stream_no, const Scanline& scanline, int num_blocks) = 0;
    
protected:
    typedef cufftComplex complex;
    
    std::vector<CudaStreamRAII::s_ptr>                  m_stream_wrappers;

    ScanSequence::s_ptr                                 m_scan_seq;
    ExcitationSignal                                    m_excitation;

    // always times equal to the number of scatterers in device memory
    size_t                                              m_num_scatterers;

    // number of samples in the time-projection lines [should be a power of two]
    size_t                                              m_num_time_samples;

    // The cuFFT plan used for all transforms.
    CufftPlanRAII::u_ptr                                m_fft_plan;

    std::vector<DeviceBufferRAII<complex>::u_ptr>       m_device_time_proj;   
    std::vector<HostPinnedBufferRAII<std::complex<float>>::u_ptr>     m_host_rf_lines;

    // precomputed excitation FFT with Hilbert mask applied.
    DeviceBufferRAII<complex>::u_ptr                    m_device_excitation_fft;

    // the value -1 means not allocated
    int                                                 m_num_beams_allocated;
    
    // it is only possible to change CUDA device before any operations
    // that involve the GPU
    bool                                                m_can_change_cuda_device;
    
    // parameters that are comon to all GPU algorithms
    int                                                 m_param_cuda_device_no;
    int                                                 m_param_num_cuda_streams;
    int                                                 m_param_threads_per_block;
    bool                                                m_store_kernel_details;

    // Always reflects the current device in use.
    cudaDeviceProp                                      m_cur_device_prop;

    // The 3D texture used as lookup-table beam profile.
    DeviceBeamProfileRAII::u_ptr                        m_device_beam_profile;

    // TEMPORARY: Cached analytical profile data
    float   m_analytical_sigma_lat;
    float   m_analytical_sigma_ele;

    // TEMPORARY: Cached lookup profile data 
    float   m_lut_r_min;
    float   m_lut_r_max;
    float   m_lut_l_min;
    float   m_lut_l_max;
    float   m_lut_e_min;
    float   m_lut_e_max;
};
    
}   // end namespace

