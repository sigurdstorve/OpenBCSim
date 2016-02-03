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
#include <tuple>
#ifdef BCSIM_ENABLE_OPENMP
    #include <omp.h>
#endif
#include "CpuBaseAlgorithm.hpp"
#include "../to_string.hpp"
#include "../LibBCSim.hpp"
#include "../BeamConvolver.hpp"
#include "common_utils.hpp" // for compute_num_rf_samples
#include "../bspline.hpp"

namespace bcsim {

void CpuBaseAlgorithm::fixed_projection_loop(const Scanline& line, std::complex<float>* time_proj_signal, size_t num_time_samples) {

    const int num_scatterers = m_scatterers->scatterers.size();
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
        const PointScatterer& scatterer = m_scatterers->scatterers[scatterer_no];
        
        // Map the global cartesian scatterer position into the beam's local
        // coordinate system.
        vector3 temp = scatterer.pos - line.get_origin();
        float r = temp.dot(line.get_direction());       // radial component
        float l = temp.dot(line.get_lateral_dir());     // lateral component
        float e = temp.dot(line.get_elevational_dir()); // elevational component
        
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

        float scaled_ampl = m_beam_profile->sampleProfile(r,l,e)*scatterer.amplitude;
        
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

void CpuBaseAlgorithm::spline_projection_loop(const Scanline& line, std::complex<float>* time_proj_signal, size_t num_time_samples) {

    const int num_scatterers = m_scatterers->num_scatterers();
    
    // The number of control points most be at least one more than the degree
    const int num_control_points = m_scatterers->get_num_control_points();
    if (num_control_points <= m_scatterers->spline_degree) {
        throw std::runtime_error("too few spline control points for given degree");
    }
    
    std::vector<float> basis_functions(num_control_points);
    
    int mu = bspline_storve::compute_knot_interval(m_scatterers->knot_vector, line.get_timestamp());

    int lower_lim = 0;
    int upper_lim = num_control_points-1;
    if (m_param_sum_all_cs) {
        std::cout << "debug mode: summing over i = " << lower_lim << "..." << upper_lim << std::endl;
    } else {
        std::tie(lower_lim, upper_lim) = bspline_storve::get_lower_upper_inds(m_scatterers->knot_vector,
                                                                              line.get_timestamp(),
                                                                              m_scatterers->spline_degree);
        if (!sanity_check_spline_lower_upper_bound(basis_functions, lower_lim, upper_lim)) {
            throw std::runtime_error("b-spline basis bounds failed sanity check");
        }
    }

    // Precompute all B-spline basis function for current timestep
    for (int i = 0; i < num_control_points; i++) {
        const float b = bspline_storve::bsplineBasis(i,
                                                        m_scatterers->spline_degree,
                                                        line.get_timestamp(),
                                                        m_scatterers->knot_vector);
        basis_functions[i] = b;
    }
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {

        // Compute position of current scatterer by evaluating spline in current timestep        
        vector3 scatterer_pos(0.0f, 0.0f, 0.0f);
        for (int i = lower_lim; i <= upper_lim; i++) {
            scatterer_pos += m_scatterers->control_points[scatterer_no][i]*basis_functions[i];
        }
        
        // Map the global cartesian scatterer position into the beam's local
        // coordinate system.
        vector3 temp = scatterer_pos - line.get_origin();
        float r = temp.dot(line.get_direction());       // radial component
        float l = temp.dot(line.get_lateral_dir());     // lateral component
        float e = temp.dot(line.get_elevational_dir()); // elevational component
        
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
        const float sampling_time_step = 1.0/m_excitation.sampling_frequency;
        int closest_index = (int) std::floor(r*2.0/(m_param_sound_speed*sampling_time_step)+0.5f);
        
        float scaled_ampl = m_beam_profile->sampleProfile(r,l,e)*m_scatterers->amplitudes[scatterer_no];
        
        // Avoid out of bound seg.fault
        if (closest_index < 0 || closest_index >= num_time_samples) {
            continue;
        }

        if (m_enable_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = r*2.0/(m_param_sound_speed*sampling_time_step);
            const float ss_delay = (closest_index - true_index)/m_excitation.sampling_frequency;
            const float complex_phase = 6.283185307179586*m_excitation.demod_freq*ss_delay;

            // phase-delay
            time_proj_signal[closest_index] += scaled_ampl*std::exp(std::complex<float>(0.0f, complex_phase));
        } else {
            time_proj_signal[closest_index] += std::complex<float>(scaled_ampl, 0.0f);
        }
    }
}


CpuBaseAlgorithm::CpuBaseAlgorithm()
        : m_scan_sequence_configured(false),
          m_excitation_configured(false),
          m_scatterers_configured(false),
          m_omp_num_threads(1),
          m_param_sum_all_cs(false) {
    
    // use all cores by default
    set_use_all_available_cores();
}

void CpuBaseAlgorithm::set_use_all_available_cores() {
#ifdef BCSIM_ENABLE_OPENMP
    set_use_specific_num_cores(omp_get_max_threads());
#else
    set_use_specific_num_cores(1);
#endif
}

void CpuBaseAlgorithm::set_use_specific_num_cores(int num_threads) {
    int max_threads;
#ifdef BCSIM_ENABLE_OPENMP
    max_threads = omp_get_max_threads();
#else
    max_threads = 1;
#endif
    if (num_threads <= 0) {
        throw std::runtime_error("Number of cores must be at least one.");
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
    } else if (key == "sum_all_cs") {
        if ((value == "on") || (value == "true")) {
            m_param_sum_all_cs = true;
        } else if ((value == "off") || (value == "false")) {
            m_param_sum_all_cs = false;
        } else {
            throw std::runtime_error("invalid value for " + key);
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

void CpuBaseAlgorithm::simulate_lines(std::vector<std::vector<std::complex<float>> > & rfLines) {
    throw_if_not_configured();
    const auto num_scanlines = m_scan_sequence->get_num_lines();
    rfLines.resize(num_scanlines);

    if (m_param_verbose) {
        std::cout << "Sound speed: " << m_param_sound_speed << std::endl;
        std::cout << "Number of scan lines: " << num_scanlines << std::endl;
        std::cout << "Setting " << m_omp_num_threads << " OpenMP threads." << std::endl;
        std::cout << "IQ demodulation frequency is " << m_excitation.demod_freq << " Hz." << std::endl;
    }    
#ifdef BCSIM_ENABLE_OPENMP
    omp_set_num_threads(m_omp_num_threads);
    #pragma omp parallel for
#endif
    for (int line_no = 0; line_no < num_scanlines; line_no++) {
        const auto& line = m_scan_sequence->get_scanline(line_no);
        if (m_param_verbose) {
            std::cout << "Line " << line_no << ":\n";
        }
        rfLines[line_no] = simulate_line(line);
    }
}

std::vector<std::complex<float>> CpuBaseAlgorithm::simulate_line(const Scanline& line) {
#ifdef BCSIM_ENABLE_OPENMP
    const int thread_idx = omp_get_thread_num();
#else
    const int thread_idx = 0;
#endif

    if (m_param_verbose) {
        std::cout << " thread id:" << thread_idx << "...\n";
    }
    
    // this will have length num_time_samples [which is valid before padding starts]
    auto time_proj_signal = convolvers[thread_idx]->get_zeroed_time_proj_signal();

    // Implementation differs depending on scatterers model.
    projection_loop(line, time_proj_signal, m_rf_line_num_samples);

#ifdef BCSIM_ENABLE_NAN_CHECK
    for (size_t i = 0; i < m_rf_line_num_samples; i++) {
        // NOTE: will probably not work if compile with "fast-math", so it makes
        // most sense to do this check for debug builds.
        if (time_proj_signal[i] != time_proj_signal[i])  {
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
    std::vector<std::complex<float>> res;
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
    if (!m_scan_sequence_configured) {
        throw std::runtime_error("Scan sequence not configured.");
    }
    if (!m_excitation_configured) {
        throw std::runtime_error("Excitation not configured.");
    }
    if (m_cur_beam_profile_type == BeamProfileType::NOT_CONFIGURED){
        throw std::runtime_error("Beam profile not configured.");
    }
    if (!m_scatterers_configured) {
        throw std::runtime_error("Scatterers not configured.");
    }
}

void CpuBaseAlgorithm::set_analytical_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting analytical beam profile for CPU algorithm" << std::endl;

    const auto temp = std::dynamic_pointer_cast<GaussianBeamProfile>(beam_profile);
    if (!temp) throw std::runtime_error("CpuBaseAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfileType::ANALYTICAL;

    m_beam_profile = beam_profile;
}

void CpuBaseAlgorithm::set_lookup_profile(IBeamProfile::s_ptr beam_profile) {
    std::cout << "Setting LUT beam profile for CPU algorithm" << std::endl;

    const auto temp = std::dynamic_pointer_cast<LUTBeamProfile>(beam_profile);
    if (!temp) throw std::runtime_error("CpuBaseAlgorithm: failed to cast beam profile");
    m_cur_beam_profile_type = BeamProfileType::LOOKUP;

    m_beam_profile = beam_profile;
}

void CpuBaseAlgorithm::clear_fixed_scatterers() {
    m_scatterers_collection.fixed_collections.clear();
}

void CpuBaseAlgorithm::add_fixed_scatterers(FixedScatterers::s_ptr fixed_scatterers) {
    m_scatterers_collection.fixed_collections.push_back(fixed_scatterers);
}

void CpuBaseAlgorithm::clear_spline_scatterers() {
    m_scatterers_collection.spline_collections.clear();
}

void CpuBaseAlgorithm::add_spline_scatterers(SplineScatterers::s_ptr spline_scatterers) {
    m_scatterers_collection.spline_collections.push_back(spline_scatterers);
}

}   // namespace

