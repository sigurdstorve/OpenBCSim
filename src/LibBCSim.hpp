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
#include <memory>
#include <vector>
#include "export_macros.hpp"
#include "bcsim_defines.h"
#include "BCSimConfig.hpp"
#include "ScanSequence.hpp"
#include "BeamProfile.hpp"

namespace bcsim {

// Interface for simulator algorithm implementations
class IAlgorithm {
public:
    typedef std::shared_ptr<IAlgorithm> s_ptr;
    typedef std::unique_ptr<IAlgorithm> u_ptr;
    
    virtual ~IAlgorithm() { }
            
    // Set misc. parameters. Available keys depends on the algorithm.
    virtual void set_parameter(const std::string&, const std::string& value)        = 0;
    
    // Configure the scatterers used when simulating.
    virtual void set_scatterers(Scatterers::s_ptr new_scatterers)               = 0;

    // Set scan sequence to use when simulating all RF lines.
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)           = 0;

    // Set the excitation signal to use when convolving.
    virtual void set_excitation(const ExcitationSignal& new_excitation)             = 0;

    // Set the beam profile object to use when simulating.
    virtual void set_beam_profile(IBeamProfile::s_ptr beam_profile)                 = 0;

    /* Simulate all RF lines. Returns vector of RF lines.
       Requires that everything is properly configured. */
    virtual void simulate_lines(std::vector<std::vector<bc_float> >&  /*out*/ rf_lines) = 0;
};

// Factory function for creating simulator instances.
// Valid types are:
//     "fixed"       - Using fixed set of point scatterers.
//     "spline"      - Using spline trajectories for point scatterers.
//     "gpu_fixed"   - GPU implementation of the fixed-scatterer algorithm
//     "gpu_spline1" - GPU implementation of the spline-scatterer algorithm,
//                     with the restriction that all scanlines in the scan
//                     sequence must have the same timestamp.
//     "gpu_spline2" - General GPU implmentation of the spline algorithm,
//                     where all splines are rendered in the projection
//                     kernel (supports different timestamps in scanseq)
IAlgorithm::s_ptr DLL_PUBLIC Create(const std::string& sim_type);

}   // namespace
