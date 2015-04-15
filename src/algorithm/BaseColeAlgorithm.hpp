#pragma once
#include <vector>
#include "cole_defines.h"
#include "ColeConfig.hpp"
#include "ScanSequence.hpp"
#include "BeamProfile.hpp"
#include "FFTConvolver.hpp"
#include "LibCole.hpp"

namespace cole {

// Implementation of all common stuff.
class BaseColeAlgorithm : public IColeAlgorithm {
public:
    BaseColeAlgorithm();
    
    // Control printing of current line number being simulated.
    virtual void set_verbose(bool v);
    
    // Set general simulation parameters from a struct.
    virtual void set_parameters(const SimulationParams& new_params);
    
    // Set scan sequence to use when simulating all RF lines.
    virtual void set_scan_sequence(const ScanSequence& new_scan_sequence);

    // Set the excitation signal to use when convolving.
    virtual void set_excitation(const ExcitationSignal& new_excitation);

    // Set the beam profile object to use when simulating.
    virtual void set_beam_profile(IBeamProfile::s_ptr beam_profile);

    // Simulate all RF lines. Returns vector of RF lines.
    // Requires that everything is properly configured.
    virtual void simulate_lines(std::vector<std::vector<cole_float> >&  /*out*/ rf_lines);

    // Use as many cores as possible for simulation.
    virtual void set_use_all_available_cores();
    
    // Use a specific number of cores for simulation.
    virtual void set_use_specific_num_cores(int num_cores);

    virtual ScanSequence get_scan_sequence() const {
        return m_scan_sequence;
    }
    
protected:
    // Configure the convolvers to reflect the parameter settings
    // if all relevant have values.
    // Delete any old convolvers and create new which reflects the
    // possibly changed result size.
    void configure_convolvers_if_possible();
    
    // Throw a runtime_error if everything isn't properly configured.
    void throw_if_not_configured();

    // Get the number of samples that we will get when we simulate
    // the RF line with current parameter settings.
    int get_rf_line_num_samples(const Scanline& line);
    
    // Get the number of samples the convolvers must be configured
    // to output prior to multithreaded usage.
    int get_common_convolution_output_length();

    // Simulate a single RF line.
    // Returns a std::vector of RF signal samples.
    // Sampling frequency is the same as for the excitation signal.
    std::vector<cole_float> simulate_line(const Scanline& line);

    // Main projection loop must be implemented in subclass.
    virtual void projection_loop(const Scanline& line, std::vector<cole_float>& time_proj_signal) = 0;

protected:
    // General simulation parameters.
    SimulationParams                         m_params;        
    // Geometry of all lines to be simulated in a frame.
    ScanSequence                             m_scan_sequence;
    // Excitation RF signal.
    ExcitationSignal                         m_excitation;
    // Pointer to one FFT-convolver for each thread.
    std::vector<FFTConvolver<cole_float>::s_ptr > convolvers;
    // The beam profile (analytical expression or LUT)
    IBeamProfile::s_ptr                      m_beamProfile;
    
    // Configuration flags needed to ensure everything is configured
    // before doing the simulations.
    bool m_params_configured;
    bool m_scan_sequence_configured;
    bool m_excitation_configured;
    bool m_beam_profile_configured; 
    bool m_scatterers_configured;   
    
    // Print status information e.g. current line no
    bool m_verbose;
    // Number of threads to use for simulation.
    int  m_omp_num_threads;
};

}   // namespace

