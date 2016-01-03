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
#include <random>
#include "bcsim_defines.h"
#include "BCSimConfig.hpp"
#include "ScanSequence.hpp"
#include "BeamProfile.hpp"
#include "BeamConvolver.hpp"
#include "BaseAlgorithm.hpp"

namespace bcsim {

// Implementation common functionality for the CPU-based algorithms.
class CpuBaseAlgorithm : public BaseAlgorithm {
public:
    CpuBaseAlgorithm();
        
    virtual void set_parameter(const std::string& key, const std::string& value)        override;
    
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)               override;

    virtual void set_excitation(const ExcitationSignal& new_excitation)                 override;

    virtual void set_beam_profile(IBeamProfile::s_ptr beam_profile)                     override;

    virtual void simulate_lines(std::vector<std::vector<std::complex<bc_float>> >&  /*out*/ rf_lines) override;
    
protected:
    // Use as many cores as possible for simulation.
    void set_use_all_available_cores();
    
    // Use a specific number of cores for simulation.
    void set_use_specific_num_cores(int num_cores);

    // Configure the convolvers to reflect the parameter settings
    // if all relevant have values.
    // Delete any old convolvers and create new which reflects the
    // possibly changed result size.
    void configure_convolvers_if_possible();
    
    // Throw a runtime_error if everything isn't properly configured.
    void throw_if_not_configured();
    
    // Simulate a single RF line.
    // Returns a std::vector of RF signal samples.
    // Sampling frequency is the same as for the excitation signal.
    std::vector<bc_float> simulate_line(const Scanline& line);

    // Main projection loop must be implemented in subclass.
    virtual void projection_loop(const Scanline& line, double* time_proj_signal, size_t num_time_samples) = 0;

protected:
    // Geometry of all lines to be simulated in a frame.
    ScanSequence::s_ptr                      m_scan_sequence;
    // Excitation RF signal.
    ExcitationSignal                         m_excitation;
    // Pointer to one FFT-convolver for each thread.
    std::vector<IBeamConvolver::ptr>         convolvers;
    // The beam profile (analytical expression or LUT)
    IBeamProfile::s_ptr                      m_beamProfile;
    
    // The number of time samples in each RF line in the scan sequence.
    size_t                                  m_rf_line_num_samples;

    // Configuration flags needed to ensure everything is configured
    // before doing the simulations.
    bool m_scan_sequence_configured;
    bool m_excitation_configured;
    bool m_beam_profile_configured; 
    bool m_scatterers_configured;   
    
    // Number of threads to use for simulation.
    int  m_omp_num_threads;

    // Amplitude of Gaussian noise that will be added to the time-projected
    // signal prior to convolution.
    std::default_random_engine      m_random_engine;
    std::normal_distribution<float> m_normal_dist;
};

}   // end namespace

