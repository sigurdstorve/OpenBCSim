#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include "cole_defines.h"
#include "BaseColeAlgorithm.hpp"
#include "to_string.hpp"
#include "LibCole.hpp"
#include "FixedColeAlgorithm.hpp"
#include "SplineColeAlgorithm.hpp"
#include "safe_omp.h"
#if COLE_ENABLE_NAN_CHECK
    #include <boost/math/special_functions/fpclassify.hpp>
#endif

namespace cole {

IColeAlgorithm::s_ptr Create(const std::string& cole_type) {
    if (cole_type == "fixed") {
        return IColeAlgorithm::s_ptr(new FixedColeAlgorithm);
    } else if (cole_type == "spline") {
        return IColeAlgorithm::s_ptr(new SplineColeAlgorithm);
    } else {
        throw std::runtime_error("Illegal algorithm type: " + cole_type);
    }
}

BaseColeAlgorithm::BaseColeAlgorithm()
        : m_params_configured(false),
          m_scan_sequence_configured(false),
          m_excitation_configured(false),
          m_verbose(false),
          m_beam_profile_configured(false),
          m_scatterers_configured(false),
          m_omp_num_threads(1) {
    
    // We want as much speed as possible by default.
    set_use_all_available_cores();

}

void BaseColeAlgorithm::set_use_all_available_cores() {
    set_use_specific_num_cores(omp_get_max_threads());
}

void BaseColeAlgorithm::set_use_specific_num_cores(int num_threads) {
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
    
    // If we changed number of cores of a already configured object,
    // we must make sure the number of FFT convolvers match.
    configure_convolvers_if_possible();
}


void BaseColeAlgorithm::set_verbose(bool v) {
    m_verbose = v;
}

void BaseColeAlgorithm::set_parameters(const SimulationParams& new_params) {
    m_params = new_params;
    m_params_configured = true;
    configure_convolvers_if_possible();
}

void BaseColeAlgorithm::set_scan_sequence(const ScanSequence& new_scan_sequence) {
    m_scan_sequence = new_scan_sequence;
    m_scan_sequence_configured = true;
    configure_convolvers_if_possible();
}


void BaseColeAlgorithm::set_excitation(const ExcitationSignal& new_excitation) {
    m_excitation = new_excitation;
    m_excitation_configured = true;
    configure_convolvers_if_possible();
}   

void BaseColeAlgorithm::set_beam_profile(IBeamProfile::s_ptr beam_profile) {
    m_beamProfile = beam_profile;
    m_beam_profile_configured = true;
}

void BaseColeAlgorithm::simulate_lines(std::vector<std::vector<cole_float> > & rfLines) {
    throw_if_not_configured();
    int num_scanlines = m_scan_sequence.get_num_lines();
    rfLines.resize(num_scanlines);

    if (m_verbose) {
        std::cout << to_string(m_params) << std::endl;
        std::cout << "Number of scan lines: " << num_scanlines << std::endl;
    }
    if (m_verbose) std::cout << "Setting " << m_omp_num_threads << " OpenMP threads." << std::endl;
    
    omp_set_num_threads(m_omp_num_threads);
#if COLE_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for (int line_no = 0; line_no < num_scanlines; line_no++) {
        if (m_verbose) std::cout << "Line " << line_no << " thread id:" << omp_get_thread_num() << "...\n";
        const Scanline& line = m_scan_sequence.get_scanline(line_no);
        rfLines[line_no] = simulate_line(line);
    }
}

std::vector<cole_float> BaseColeAlgorithm::simulate_line(const Scanline& line) {
    const int num_time_samples = get_rf_line_num_samples(line);
    std::vector<cole_float> time_proj_signal(num_time_samples);

    // Implementation differs depending on what kind of scatterers model we use.
    projection_loop(line, time_proj_signal);

#if COLE_ENABLE_NAN_CHECK
    for (size_t i = 0; i < time_proj_signal.size(); i++) {
        if (boost::math::isnan(time_proj_signal[i]))  {
            throw std::runtime_error("Nan in scatterer projection.");
        }
    }
#endif

    // Get the convolver associated with this thread and do lightning-fast
    // FFT convolution.
    FFTConvolver<cole_float>::s_ptr convolver = convolvers[omp_get_thread_num()];
    std::vector<cole_float> rf_line = convolver->conv(time_proj_signal, m_excitation.samples);

    // Compensate for time delay in RF signals due to the excitation
    // signal having time zero at the middle of its samples.
    std::vector<cole_float>::const_iterator first_it = rf_line.begin() + m_excitation.center_index;
    std::vector<cole_float>::const_iterator last_it = rf_line.end();
    std::vector<cole_float> return_value(first_it, last_it);
    
    return return_value;
}

void BaseColeAlgorithm::configure_convolvers_if_possible() {
    if (m_scan_sequence_configured && m_params_configured && m_excitation_configured) {
        int common_output_size = get_common_convolution_output_length();
        // TODO: Only create new if output size differs
        
        convolvers.clear();
        for (int i = 0; i < m_omp_num_threads; i++) {
            if (m_verbose) std::cout << "Creating FFT-convolver " << i << std::endl;
 
            auto convolver = FFTConvolver<cole_float>::s_ptr(new FFTConvolver<cole_float>(common_output_size));
            convolvers.push_back(convolver);
            std::cout << "Creating convolvers\nNew number " << convolvers.size() << std::endl;
        
        }
    }
}

void BaseColeAlgorithm::throw_if_not_configured() {
    if (!m_params_configured) throw std::runtime_error("Parameters not configured.");
    if (!m_scan_sequence_configured) throw std::runtime_error("Scan sequence not configured.");
    if (!m_excitation_configured) throw std::runtime_error("Excitation not configured.");
    if (!m_beam_profile_configured) throw std::runtime_error("Beam profile not configured.");
    if (!m_scatterers_configured) throw std::runtime_error("Scatterers not configured.");
}

int BaseColeAlgorithm::get_rf_line_num_samples(const Scanline& line) {
    cole_float max_time = 2.0*line.length/m_params.sound_speed;
    cole_float sampling_time_step = 1.0/m_excitation.sampling_frequency; 
    int num_time_samples = static_cast<int>(std::floor(max_time / sampling_time_step + 0.5f)); 
    return num_time_samples;
}

int BaseColeAlgorithm::get_common_convolution_output_length() {
    int num_scanlines = m_scan_sequence.get_num_lines();     
    if (num_scanlines == 0) {
        throw std::runtime_error("No scanlines.");
    }
    int common_num_samples = get_rf_line_num_samples(m_scan_sequence.get_scanline(0));
    for (int line_no = 0; line_no < num_scanlines; line_no++) {
        int temp = get_rf_line_num_samples(m_scan_sequence.get_scanline(line_no));
        if (temp != common_num_samples) {
            throw std::runtime_error("Different convolution lengths not yet implemented.");
        }
    }
    // And take the length of the excitation signal into accountusing
    // the well-known convolution length formula N1+N2-1
    common_num_samples += m_excitation.samples.size() - 1;
    return common_num_samples;
}

}   // namespace

