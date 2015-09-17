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
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "bcsim_defines.h"
#include "BaseAlgorithm.hpp"
#include "to_string.hpp"
#include "LibBCSim.hpp"
#include "safe_omp.h"
#include "SignalProcessing.hpp" // for env()
#include "BeamConvolver.hpp"
#include "common_utils.hpp" // for compute_num_rf_samples
#if BCSIM_ENABLE_NAN_CHECK
    #include <boost/math/special_functions/fpclassify.hpp>
#endif

namespace bcsim {

BaseAlgorithm::BaseAlgorithm()
        : m_sound_speed(1540.0f),
          m_scan_sequence_configured(false),
          m_excitation_configured(false),
          m_verbose(false),
          m_beam_profile_configured(false),
          m_scatterers_configured(false),
          m_omp_num_threads(1),
          m_convolver_type("rf") {
    
    // use all cores by default
    set_use_all_available_cores();
    set_noise_amplitude(0.0f);
}

void BaseAlgorithm::set_use_all_available_cores() {
    set_use_specific_num_cores(omp_get_max_threads());
}

void BaseAlgorithm::set_use_specific_num_cores(int num_threads) {
    int max_threads = omp_get_max_threads();
    if (num_threads < 0) {
        throw std::runtime_error("Number of cores cannot be negative");
    } else if (num_threads > max_threads) {
        throw std::runtime_error("Number of cores not supported by computer");
    }
    m_omp_num_threads = num_threads;
    if (m_verbose) {
        std::cout << "Using " << m_omp_num_threads << " OpenMP threads." << std::endl;
    }
    
    // number of convolvers must match number of threads
    configure_convolvers_if_possible();
}


void BaseAlgorithm::set_verbose(bool v) {
    m_verbose = v;
}

void BaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "sound_speed") {
        m_sound_speed = std::stof(value);
        // must also update convolvers.
        configure_convolvers_if_possible();
    } else {
        throw std::runtime_error("illegal parameter name");
    }
}

void BaseAlgorithm::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    if (!new_scan_sequence->is_valid()) {
        throw std::runtime_error("scan sequence is invalid");
    }
    
    if (!m_excitation_configured) {
        throw std::runtime_error("Excitation must be configured before scan sequence");        
    }
    
    m_scan_sequence = new_scan_sequence;
    m_scan_sequence_configured = true;

    const auto line_length = m_scan_sequence->line_length;
    m_rf_line_num_samples = compute_num_rf_samples(m_sound_speed, line_length, m_excitation.sampling_frequency);

    configure_convolvers_if_possible();
}


void BaseAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
    m_excitation = new_excitation;
    m_excitation_configured = true;
    configure_convolvers_if_possible();
}   

void BaseAlgorithm::set_beam_profile(IBeamProfile::s_ptr beam_profile) {
    m_beamProfile = beam_profile;
    m_beam_profile_configured = true;
}

void BaseAlgorithm::set_output_type(const std::string& type) {
    if (type == "rf") {
        m_convolver_type = "rf";
    } else if (type == "env") {
        m_convolver_type = "env";
    } else if (type == "proj") {
        m_convolver_type = "proj";
    } else {
        throw std::runtime_error("Illegal output type: " + type);
    }
}

void BaseAlgorithm::simulate_lines(std::vector<std::vector<bc_float> > & rfLines) {
    throw_if_not_configured();
    int num_scanlines = m_scan_sequence->get_num_lines();
    rfLines.resize(num_scanlines);

    if (m_verbose) {
        std::cout << "Sound speed: " << m_sound_speed << std::endl;
        std::cout << "Number of scan lines: " << num_scanlines << std::endl;
    }
    if (m_verbose) std::cout << "Setting " << m_omp_num_threads << " OpenMP threads." << std::endl;
    
    omp_set_num_threads(m_omp_num_threads);
#if BCSIM_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for (int line_no = 0; line_no < num_scanlines; line_no++) {
        if (m_verbose) std::cout << "Line " << line_no << " thread id:" << omp_get_thread_num() << "...\n";
        const Scanline& line = m_scan_sequence->get_scanline(line_no);
        rfLines[line_no] = simulate_line(line);
    }
}

std::vector<bc_float> BaseAlgorithm::simulate_line(const Scanline& line) {
    const int thread_idx       = omp_get_thread_num();
    
    // this will have length num_time_samples [which is valid before padding starts]
    double* time_proj_signal = convolvers[thread_idx]->get_zeroed_time_proj_signal();

    // Implementation differs depending on scatterers model.
    projection_loop(line, time_proj_signal, m_rf_line_num_samples);

#if BCSIM_ENABLE_NAN_CHECK
    for (size_t i = 0; i < m_rf_line_num_samples; i++) {
        if (boost::math::isnan(time_proj_signal[i]))  {
            throw std::runtime_error("Nan in scatterer projection.");
        }
    }
#endif

    // add Gaussian noise if desirable.
    if (m_noise_amplitude > 0.0f) {
        std::transform(time_proj_signal, time_proj_signal + m_rf_line_num_samples, time_proj_signal, [&](float v) {
            return v + m_normal_dist(m_random_engine);
        });
    }

    // get the convolver associated with this thread and do FFT-based convolution
    std::vector<bc_float> rf_line = convolvers[thread_idx]->process();
    return rf_line;
}

void BaseAlgorithm::configure_convolvers_if_possible() {
    if (m_scan_sequence_configured && m_excitation_configured) {
        convolvers.clear();
        std::cout << "Recreating convolvers\n";
        for (int i = 0; i < m_omp_num_threads; i++) {
            if (m_verbose) std::cout << "Creating FFT-convolver " << i << " of type " << m_convolver_type << std::endl;
 
            auto convolver = IBeamConvolver::Create(m_convolver_type, m_rf_line_num_samples, m_excitation);
            convolvers.push_back(std::move(convolver));
        }
    }
}

void BaseAlgorithm::throw_if_not_configured() {
    if (!m_scan_sequence_configured)    throw std::runtime_error("Scan sequence not configured.");
    if (!m_excitation_configured)       throw std::runtime_error("Excitation not configured.");
    if (!m_beam_profile_configured)     throw std::runtime_error("Beam profile not configured.");
    if (!m_scatterers_configured)       throw std::runtime_error("Scatterers not configured.");
}

}   // namespace

