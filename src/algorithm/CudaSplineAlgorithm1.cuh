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
#include "CudaFixedAlgorithm.cuh"

// NOTE: There is no support for double here!!!
// NOTE2: This is experimental code.

namespace bcsim {

class CudaSplineAlgorithm1 : public IAlgorithm {
public:
    CudaSplineAlgorithm1();

    virtual ~CudaSplineAlgorithm1() {
        // cleanup
    }
        
    virtual void set_verbose(bool v) {
        m_fixed_alg->set_verbose(v);
    }
    
    virtual void set_parameters(const SimulationParams& new_params) {
        m_fixed_alg->set_parameters(new_params);
    }
    
    virtual void set_scatterers(Scatterers::s_ptr new_scatterers);
    
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence);

    virtual void set_excitation(const ExcitationSignal& new_excitation) {
        m_fixed_alg->set_excitation(new_excitation);
    }

    virtual void set_beam_profile(IBeamProfile::s_ptr beam_profile) {
        m_fixed_alg->set_beam_profile(beam_profile);
    }

    virtual void set_output_type(const std::string& type) {
        if (type == "env") {
            // this is hardcoded for now
        } else if (type == "rf") {
            throw std::runtime_error("Output data type 'rf' is not yet supported");
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

private:
    // Test if all scanlines in a scan sequence have the same timestamp
    bool has_equal_timestamps(ScanSequence::s_ptr scan_seq, double tol=1e-4);

private:

    // An internal instance of the fixed simulator algorithm.
    std::shared_ptr<CudaFixedAlgorithm>     m_fixed_alg;

    // GPU memory for holding the splines
    DeviceBufferRAII<float>::u_ptr              m_control_xs;
    DeviceBufferRAII<float>::u_ptr              m_control_ys;
    DeviceBufferRAII<float>::u_ptr              m_control_zs;
    DeviceBufferRAII<float>::u_ptr              m_control_as;

    // The knot vector common to all splines.
    std::vector<float>                          m_common_knots;
    int                                         m_num_cs;
    int                                         m_spline_degree;
    int                                         m_num_splines;
};

}   // end namespace