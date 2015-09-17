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
#include "LibBCSim.hpp"
#include "cuda_helpers.h"
#include "cufft_helpers.h"

// NOTE: There is no support for double here!!!

namespace bcsim {

class CudaSplineAlgorithm2 : public IAlgorithm {
public:

    CudaSplineAlgorithm2();

    virtual ~CudaSplineAlgorithm2() {
        // cleanup
    }
        
    virtual void set_verbose(bool v) {
        m_verbose = v;
    }
    
    virtual void set_parameter(const std::string& key, const std::string& value) {
        if (key == "sound_speed") {
            m_sound_speed = std::stof(value);
        } else {
            throw std::runtime_error("illegal parameter key");
        }
    }
    
    virtual void set_scatterers(Scatterers::s_ptr new_scatterers);
    
    // NOTE: currently requires that set_excitation is called first!
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence);

    virtual void set_excitation(const ExcitationSignal& new_excitation);

    virtual void set_beam_profile(IBeamProfile::s_ptr beam_profile) {
        auto gaussian_profile = std::dynamic_pointer_cast<bcsim::GaussianBeamProfile>(beam_profile);
        if (!gaussian_profile) {
            throw std::runtime_error("GPU algorithm currently only supports analytical beam profiles");
        }
        m_beam_profile = gaussian_profile;   
    }

    // NOTE: currently requires that set_excitation is called first!
    virtual void set_output_type(const std::string& type) {
        if (type == "env") {
            m_output_type = "env";
        } else if (type == "rf") {
            m_output_type = "rf";
        } else if (type == "proj") {
            throw std::runtime_error("Output data type 'proj' is not yet supported");
        }
    }

    virtual void simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines);
    
    virtual void set_use_all_available_cores() {
        // does not apply in the GPU case.
    }
    
    virtual void set_use_specific_num_cores(int numCores) {
        // does not apply in the GPU case.
    }

    virtual void set_noise_amplitude(float amplitude) {
        // does not apply in the GPU case (yet)
    }

protected:
    void copy_scatterers_to_device(SplineScatterers::s_ptr scatterers);

protected:
    typedef cufftComplex complex;

    // the output data type
    std::string             m_output_type;

    ScanSequence::s_ptr     m_scan_seq;
    float                   m_sound_speed;
    bool                    m_verbose;
    ExcitationSignal        m_excitation;

    // At all times equal to the number of scatterers in device memory
    size_t                  m_num_scatterers;

    // number of samples in the time-projection lines [should be a power of two]
    size_t              m_num_time_samples;

    std::vector<CudaStreamRAII::u_ptr>  m_stream_wrappers;
    
    // device memory for spline scatterers control points and amplitudes
    DeviceBufferRAII<float>::u_ptr      m_device_control_xs;
    DeviceBufferRAII<float>::u_ptr      m_device_control_ys;
    DeviceBufferRAII<float>::u_ptr      m_device_control_zs;
    DeviceBufferRAII<float>::u_ptr      m_device_control_as;
    
    // The cuFFT plan used for all transforms.
    CufftPlanRAII::u_ptr                m_fft_plan;

    std::vector<DeviceBufferRAII<float>::u_ptr>            m_device_time_proj;    // real-valued
    std::vector<DeviceBufferRAII<complex>::u_ptr>          m_device_rf_lines;     // complex-valued
    std::vector<DeviceBufferRAII<float>::u_ptr>            m_device_rf_lines_env; // real-valued
    std::vector<HostPinnedBufferRAII<float>::u_ptr>        m_host_rf_lines;       // real-valued

    // precomputed excitation FFT with Hilbert mask applied.
    DeviceBufferRAII<complex>::u_ptr                  m_device_excitation_fft;

    // -1 means not allocated
    int     m_num_beams_allocated;

    // TODO: Figure out how to support LUT beam profiles also.
    std::shared_ptr<bcsim::GaussianBeamProfile>  m_beam_profile;
    
    // The knot vector common to all splines.
    std::vector<float>                          m_common_knots;
    int                                         m_num_cs;
    int                                         m_spline_degree;
    int                                         m_num_splines;
};

}   // end namespace