#include <iostream>
#include <vector>
#include <stdexcept>
#include "cole_defines.h"
#include "FixedColeAlgorithm.hpp"
#include "safe_omp.h"

namespace cole {

FixedColeAlgorithm::FixedColeAlgorithm()
    : BaseColeAlgorithm() { }

void FixedColeAlgorithm::set_scatterers(ColeScatterers::s_ptr new_scatterers) {
    m_scatterers = std::dynamic_pointer_cast<ColeFixedScatterers>(new_scatterers);
    
    if (m_scatterers == nullptr) {
        throw std::runtime_error("Cast to ColeFixedScatterers failed");
    }
    
    if (m_verbose) std::cout << "Number of spline scatterers: " << m_scatterers->num_scatterers() << std::endl;
    m_scatterers_configured = true;
}

void FixedColeAlgorithm::projection_loop(const Scanline& line, std::vector<cole_float>& time_proj_signal) {

    // TODO: Use get() to avoid overhead of smart pointers in inner loop?
    // Back up by benchmark results!

    const int num_time_samples = time_proj_signal.size();
    const int num_scatterers = m_scatterers->scatterers.size();
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
        const PointScatterer& scatterer = m_scatterers->scatterers[scatterer_no];
        
        // Map the global cartesian scatterer position into the beam's local
        // coordinate system.
        vec3 temp = scatterer.pos - line.origin;
        cole_float r = temp.dot(line.direction);             // radial component
        cole_float l = temp.dot(line.lateral_dir);           // lateral component
        cole_float e = temp.dot(line.get_elevational_dir()); // elevational component
        
        // Add scaled amplitude to closest index
        const cole_float sampling_time_step = 1.0/m_excitation.sampling_frequency;
        int closest_index = (int) std::floor(r*2.0/(m_params.sound_speed*sampling_time_step)+0.5f);
        
        cole_float scaled_ampl = m_beamProfile->sampleProfile(r,l,e)*scatterer.amplitude;
        
        // Avoid out of bound seg.fault
        if (closest_index < 0 || closest_index >= num_time_samples) {
#if COLE_PRINT_SCATTERER_OUTSIDE_MSG
            std::cout << "[warning] scatterer outside scan line." << std::endl;
#endif
            continue;
        }
        time_proj_signal[closest_index] += scaled_ampl;
    }
}


}   // namespace
