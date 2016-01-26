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

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "CpuFixedAlgorithm.hpp"
#ifdef BCSIM_ENABLE_OPENMP
    #include <omp.h>
#endif

namespace bcsim {

CpuFixedAlgorithm::CpuFixedAlgorithm()
    : CpuBaseAlgorithm() { }

void CpuFixedAlgorithm::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_scatterers = std::dynamic_pointer_cast<FixedScatterers>(new_scatterers);
    
    if (m_scatterers == nullptr) {
        throw std::runtime_error("Cast to FixedScatterers failed");
    }
    
    if (m_param_verbose) std::cout << "Number of fixed scatterers: " << m_scatterers->num_scatterers() << std::endl;
    m_scatterers_configured = true;
}

void CpuFixedAlgorithm::projection_loop(const Scanline& line, std::complex<float>* time_proj_signal, size_t num_time_samples) {

    const int num_scatterers = m_scatterers->scatterers.size();
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
        const PointScatterer& scatterer = m_scatterers->scatterers[scatterer_no];
        
        // Map the global cartesian scatterer position into the beam's local
        // coordinate system.
        vector3 temp = scatterer.pos - line.get_origin();
        bc_float r = temp.dot(line.get_direction());       // radial component
        bc_float l = temp.dot(line.get_lateral_dir());     // lateral component
        bc_float e = temp.dot(line.get_elevational_dir()); // elevational component
        
        // Use "arc projection" in the radial direction: use length of vector from
        // beam's origin to the scatterer with the same sign as the projection onto
        // the line.
        if (m_param_use_arc_projection) {
#ifdef __GNUC__
            r = std::copysign(temp.norm(), r);
#else
            r = _copysignf(temp.norm(), r);
#endif            
        }
        
        // Add scaled amplitude to closest index
        int closest_index = (int) std::floor(r*2.0*m_excitation.sampling_frequency/(m_param_sound_speed)+0.5f);

        bc_float scaled_ampl = m_beam_profile->sampleProfile(r,l,e)*scatterer.amplitude;
        
        // Avoid out of bound seg.fault
        if (closest_index < 0 || closest_index >= num_time_samples) {
            continue;
        }


        if (m_enable_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = r*2.0*m_excitation.sampling_frequency/(m_param_sound_speed);
            const float ss_delay = (closest_index - true_index)/m_excitation.sampling_frequency;
            const float complex_phase = 6.283185307179586*m_excitation.demod_freq*ss_delay;

            // phase-delay
            time_proj_signal[closest_index] += scaled_ampl*std::exp(std::complex<float>(0.0f, complex_phase));
        } else {
            time_proj_signal[closest_index] += std::complex<float>(scaled_ampl, 0.0f);
        }
    }
}


}   // namespace
