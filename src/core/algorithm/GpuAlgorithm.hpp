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
#pragma once
#include <vector>
#include <memory>
#include <cuda.h>
#include <cufft.h>
#include "cuda_helpers.h"
#include "cufft_helpers.h"
#include "BaseAlgorithm.hpp"
#include "GpuScatterers.hpp"
#include "curand_helpers.h"

namespace bcsim {

// convert bcsim::vector3 to CUDA float3 datatype
inline float3 to_float3(const bcsim::vector3& v) {
    return make_float3(v.x, v.y, v.z);
}

class GpuAlgorithm : public BaseAlgorithm {
public:
    GpuAlgorithm();

    virtual ~GpuAlgorithm() {
        // TODO: Somehow call cudaDeviceReset() without crashes that
        // occur most likely when RAII-wrappers go out of scope and
        // tries to free CUDA resources..
    }
    
    virtual void set_parameter(const std::string& key, const std::string& value)        override;

    virtual std::string get_parameter(const std::string& key) const                     override;

    virtual void simulate_lines(std::vector<std::vector<std::complex<float>> >&  /*out*/ rf_lines) override;
    
    // NOTE: currently requires that set_excitation is called first!
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)               override;

    virtual void set_excitation(const ExcitationSignal& new_excitation)                 override;
    
    virtual void set_analytical_profile(IBeamProfile::s_ptr beam_profile) override;

    virtual void set_lookup_profile(IBeamProfile::s_ptr beam_profile) override;

    virtual void clear_fixed_scatterers()                                                           override;

    virtual void add_fixed_scatterers(FixedScatterers::s_ptr)                                       override;

    virtual void clear_spline_scatterers()                                                          override;

    virtual void add_spline_scatterers(SplineScatterers::s_ptr)                                     override;

    virtual size_t get_total_num_scatterers() const                                     override;

protected:
    // Debug functionality: slice the 3D texture and write as RAW file to disk.    
    void dump_orthogonal_lut_slices(const std::string& raw_path);

    void create_cuda_stream_wrappers(int num_streams);
    
    int get_num_cuda_devices() const;
    
    void save_cuda_device_properties();
        
    // to ensure that calls to device beam profile RAII wrapper does not cause segfault.
    void create_dummy_lut_profile();

    void fixed_projection_kernel(int stream_no, const Scanline& scanline, int num_blocks, cuComplex* res_buffer, DeviceFixedScatterers::s_ptr dataset);

    void spline_projection_kernel(int stream_no, const Scanline& scanline, int num_blocks, cuComplex* res_buffer, DeviceSplineScatterers::s_ptr dataset);

protected:
    typedef cufftComplex complex;
    
    std::vector<CudaStreamRAII::s_ptr>                  m_stream_wrappers;

    ScanSequence::s_ptr                                 m_scan_seq;
    ExcitationSignal                                    m_excitation;

    // number of samples in the time-projection lines [should be a power of two]
    size_t                                              m_num_time_samples;

    // The cuFFT plan used for all transforms.
    CufftBatchedPlanRAII::u_ptr                         m_fft_plan;

    DeviceBufferRAII<complex>::u_ptr                    m_device_time_proj;   
    std::vector<HostPinnedBufferRAII<std::complex<float>>::u_ptr>     m_host_rf_lines;

    // precomputed excitation FFT with Hilbert mask applied.
    DeviceBufferRAII<complex>::u_ptr                    m_device_excitation_fft;

    // The number of RF lines memory is allocated for, and also cuFFT batched
    // transform plan is configure for. The value -1 means not allocated yet.
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

    // TODO: set log callbacks!
    DeviceFixedScatterersCollection     m_device_fixed_datasets;
    DeviceSplineScatterersCollection    m_device_spline_datasets;

    // optimization to reduce memory bandwidth usage when all lines
    // in a scan have the same timestamp.
    DeviceFixedScatterersCollection     m_device_rendered_spline_datasets;

    CurandGeneratorRAII                 m_device_rng;
    DeviceBufferRAII<float>::u_ptr      m_device_random_buffer;
};
    
}   // end namespace

#endif  // BCSIM_ENABLE_CUDA
