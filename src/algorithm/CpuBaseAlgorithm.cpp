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
#include "CpuBaseAlgorithm.hpp"
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

CpuBaseAlgorithm::CpuBaseAlgorithm()
        : m_scan_sequence_configured(false),
          m_excitation_configured(false),
          m_scatterers_configured(false),
          m_omp_num_threads(1) {
    
    // use all cores by default
    set_use_all_available_cores();
}

void CpuBaseAlgorithm::set_use_all_available_cores() {
    set_use_specific_num_cores(omp_get_max_threads());
}

void CpuBaseAlgorithm::set_use_specific_num_cores(int num_threads) {
    int max_threads = omp_get_max_threads();
    if (num_threads < 0) {
        throw std::runtime_error("Number of cores cannot be negative");
    } else if (num_threads > max_threads) {
        throw std::runtime_error("Number of cores not supported by computer");
    }
    m_omp_num_threads = num_threads;
    if (m_param_verbose) {
        std::cout << "Using " << m_omp_num_threads << " OpenMP threads." << std::endl;
    }
    
    // number of convolvers must match number of threads
    configure_convolvers_if_possible();
}

void CpuBaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "sound_speed") {
        BaseAlgorithm::set_parameter(key, value);
        // convolvers must be updated after sound speed has changed.
        configure_convolvers_if_possible();
    } else if (key == "num_cpu_cores") {
        if (value == "all") {
            set_use_all_available_cores();
        } else {
            // TODO: catch any exceptions
            const auto num_cores = static_cast<int>(std::stoi(value));
            set_use_specific_num_cores(num_cores);
        }
    } else if (key == "noise_amplitude") { 
        BaseAlgorithm::set_parameter(key, value);
        m_normal_dist = std::normal_distribution<float>(0.0f, m_param_noise_amplitude);
    } else {
        BaseAlgorithm::set_parameter(key, value);
    }
}

void CpuBaseAlgorithm::set_scan_sequence(ScanSequence::s_ptr new_scan_sequence) {
    if (!new_scan_sequence->is_valid()) {
        throw std::runtime_error("scan sequence is invalid");
    }
    
    if (!m_excitation_configured) {
        throw std::runtime_error("Excitation must be configured before scan sequence");        
    }
    
    m_scan_sequence = new_scan_sequence;
    m_scan_sequence_configured = true;

    const auto line_length = m_scan_sequence->line_length;
    m_rf_line_num_samples = compute_num_rf_samples(m_param_sound_speed, line_length, m_excitation.sampling_frequency);

    configure_convolvers_if_possible();
}


void CpuBaseAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
    m_excitation = new_excitation;
    m_excitation_configured = true;
    configure_convolvers_if_possible();
}   

void CpuBaseAlgorithm::simulate_lines(std::vector<std::vector<std::complex<bc_float>> > & rfLines) {
    throw_if_not_configured();
    const auto num_scanlines = m_scan_sequence->get_num_lines();
    rfLines.resize(num_scanlines);

    if (m_param_verbose) {
        std::cout << "Sound speed: " << m_param_sound_speed << std::endl;
        std::cout << "Number of scan lines: " << num_scanlines << std::endl;
        std::cout << "Setting " << m_omp_num_threads << " OpenMP threads." << std::endl;
        std::cout << "IQ demodulation frequency is " << m_excitation.demod_freq << " Hz." << std::endl;
    }    
    omp_set_num_threads(m_omp_num_threads);
#if BCSIM_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for (int line_no = 0; line_no < num_scanlines; line_no++) {
        if (m_param_verbose) {
            std::cout << "Line " << line_no << " thread id:" << omp_get_thread_num() << "...\n";
        }
        const auto& line = m_scan_sequence->get_scanline(line_no);
        rfLines[line_no] = simulate_line(line);
    }
}

std::vector<std::complex<bc_float>> CpuBaseAlgorithm::simulate_line(const Scanline& line) {
    const int thread_idx       = omp_get_thread_num();
    
    // this will have length num_time_samples [which is valid before padding starts]
    auto time_proj_signal = convolvers[thread_idx]->get_zeroed_time_proj_signal();

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
    /*
    if (m_param_noise_amplitude > 0.0f) {
        std::transform(time_proj_signal, time_proj_signal + m_rf_line_num_samples, time_proj_signal, [&](float v) {
            return v + m_normal_dist(m_random_engine);
        });
    }
    */

    // get the convolver associated with this thread and do FFT-based convolution
    // complex down-shifting to form a proper IQ signal. TODO: consider precomputing the complex exponential
    auto temp_line = convolvers[thread_idx]->process();
    // TODO: Consider merging decimateion. No need to compute samples that we later discard..
    const auto f_demod = m_excitation.demod_freq;
    const float norm_f_demod = f_demod/m_excitation.sampling_frequency;
    const float TWO_PI = static_cast<float>(2.0*4.0*std::atan(1));
    for (size_t i = 0; i < temp_line.size(); i++) {
        temp_line[i] *= std::exp(-TWO_PI*std::complex<float>(0.0f, 1.0)*norm_f_demod*static_cast<float>(i));
    }
    
    // Decimate
    // TODO: Consider reserve to avoid unneeded allocations resulting from push_back()?
    std::vector<std::complex<bc_float>> res;
    for (int i = 0; i < static_cast<int>(temp_line.size()); i += m_radial_decimation) {
        res.push_back(temp_line[i]);
    }

    return res;
}

void CpuBaseAlgorithm::configure_convolvers_if_possible() {
    if (m_scan_sequence_configured && m_excitation_configured) {
        convolvers.clear();
        std::cout << "Recreating convolvers\n";
        for (int i = 0; i < m_omp_num_threads; i++) {
            if (m_param_verbose) {
                std::cout << "Creating FFT-convolver " << i << std::endl;
            }
            
            auto convolver = IBeamConvolver::Create(m_rf_line_num_samples, m_excitation);
            convolvers.push_back(std::move(convolver));
        }
    }
}

void CpuBaseAlgorithm::throw_if_not_configured() {
    if (!m_scan_sequence_configured)    throw std::runtime_error("Scan sequence not configured.");
    if (!m_excitation_configured)       throw std::runtime_error("Excitation not configured.");
    if (!m_beam_profile_configured)     throw std::runtime_error("Beam profile not configured.");
    if (!m_scatterers_configured)       throw std::runtime_error("Scatterers not configured.");
}

void CpuBaseAlgorithm::set_analytical_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting analytical beam profile for CPU algorithm" << std::endl;
    m_cur_beam_profile_type = BeamProfile::ANALYTICAL;
    // TODO
}

void CpuBaseAlgorithm::set_lookup_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting LUT beam profile for CPU algorithm" << std::endl;
    m_cur_beam_profile_type = BeamProfile::LOOKUP;
    // TODO
}

}   // namespace

