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
#include "BaseAlgorithm.hpp"
#include "../BCSimConfig.hpp"
#include "../ScanSequence.hpp"
#include "../BeamProfile.hpp"
#include "../BeamConvolver.hpp"

namespace bcsim {

// A collection or zero or more fixed and spline scatterer sets.
struct PointScattererCollection {
    std::vector<FixedScatterers::s_ptr>     fixed_collections;
    std::vector<SplineScatterers::s_ptr>    spline_collections;

    // Compute the total number of fixed scatterers.
    size_t num_fixed_scatterers() const {
        size_t num_scatterers = 0;
        for (const auto& scatterers : fixed_collections) {
            num_scatterers += scatterers->num_scatterers();
        }
        return num_scatterers;
    }

    // Compute the total number of spline scatterers.
    size_t num_spline_scatterers() const {
        size_t num_scatterers = 0;
        for (const auto& scatterers : spline_collections) {
            num_scatterers += scatterers->num_scatterers();
        }
        return num_scatterers;
    }

    // Compute the overall total number of scatterers (fixed and spline)
    size_t total_num_scatterers() const {
        return num_fixed_scatterers() + num_spline_scatterers();
    }
};

// Concrete CPU simulator implementation.
class CpuAlgorithm : public BaseAlgorithm {
public:
    CpuAlgorithm();
        
    virtual void set_parameter(const std::string& key, const std::string& value)                    override;
    
    virtual void set_scan_sequence(ScanSequence::s_ptr new_scan_sequence)                           override;

    virtual void set_excitation(const ExcitationSignal& new_excitation)                             override;

    virtual void simulate_lines(std::vector<std::vector<std::complex<float>> >&  /*out*/ rf_lines)  override;

    virtual void set_analytical_profile(IBeamProfile::s_ptr beam_profile)                           override;

    virtual void set_lookup_profile(IBeamProfile::s_ptr beam_profile)                               override;

    virtual void clear_fixed_scatterers()                                                           override;

    virtual void add_fixed_scatterers(FixedScatterers::s_ptr)                                       override;

    virtual void clear_spline_scatterers()                                                          override;

    virtual void add_spline_scatterers(SplineScatterers::s_ptr)                                     override;

protected:
    // Projection loop for a single fixed scatterer dataset.
    void fixed_projection_loop(const Scanline& line, std::complex<float>* time_proj_signal, size_t num_time_samples);
    
    // Projection loop for a single spline scatterer dataset.
    void spline_projection_loop(const Scanline& line, std::complex<float>* time_proj_signal, size_t num_time_samples);

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
    // Returns a std::vector of IQ signal samples.
    // Sampling frequency is the same as for the excitation signal. TODO: Not so with decimation...
    std::vector<std::complex<float>> simulate_line(const Scanline& line);

protected:
    // Geometry of all lines to be simulated in a frame.
    ScanSequence::s_ptr                      m_scan_sequence;
    // Excitation RF signal.
    ExcitationSignal                         m_excitation;
    // Pointer to one FFT-convolver for each thread.
    std::vector<IBeamConvolver::ptr>         convolvers;
    
    PointScattererCollection                m_scatterers_collection;
    
    // The number of time samples in each RF line in the scan sequence.
    size_t                                  m_rf_line_num_samples;

    // Configuration flags needed to ensure everything is configured
    // before doing the simulations.
    bool m_scan_sequence_configured;
    bool m_excitation_configured;
    bool m_scatterers_configured;   
    
    // Number of threads to use for simulation.
    int  m_omp_num_threads;

    // Amplitude of Gaussian noise that will be added to the time-projected
    // signal prior to convolution.
    std::default_random_engine      m_random_engine;
    std::normal_distribution<float> m_normal_dist;

    // Current active beam profile.
    IBeamProfile::s_ptr             m_beam_profile;         // TEMPORARY

    // Debug parameter: If true, sum over all B-spline basis functions instead of
    // only those with non-zero basis functions. Result should be the same.
    bool                       m_param_sum_all_cs;
};

}   // end namespace

