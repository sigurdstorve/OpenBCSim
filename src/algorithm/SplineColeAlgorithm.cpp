#include <iostream>
#include <vector>
#include <stdexcept>
#include "cole_defines.h"
#include "SplineColeAlgorithm.hpp"
#include "bspline.hpp"
#include "safe_omp.h"

namespace cole {

SplineColeAlgorithm::SplineColeAlgorithm()
        : BaseColeAlgorithm() { }
     
void SplineColeAlgorithm::set_scatterers(ColeScatterers::s_ptr new_scatterers) {
    m_scatterers = std::dynamic_pointer_cast<ColeSplineScatterers>(new_scatterers);

    if (m_scatterers == nullptr) {
        throw std::runtime_error("Cast to ColeSplineScatterers failed");
    }

    if (m_verbose) std::cout << "Number of spline scatterers: " << m_scatterers->num_scatterers() << std::endl;
    m_scatterers_configured = true;
}

void SplineColeAlgorithm::projection_loop(const Scanline& line, std::vector<cole_float>& time_proj_signal) {

    // TODO: Use get() to avoid overhead of smart pointers in inner loop?
    // Back up by benchmark results!
    const int num_time_samples = time_proj_signal.size();
    const int num_scatterers = m_scatterers->nodes.size();
    // TODO: Improve. Use that all splines have same number of control points
    const int num_control_points = m_scatterers->nodes[0].size();
    std::vector<cole_float> basis_functions(num_control_points);
    
    // Precompute all B-spline basis function for current timestep
    for (int i = 0; i < num_control_points; i++) {
        const cole_float b = bspline_storve::bsplineBasis(i,
                                                     m_scatterers->spline_degree,
                                                     line.timestamp,
                                                     m_scatterers->knot_vector);
        basis_functions[i] = b;
    }
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
    
        // Compute position of current scatterer by evaluating spline in current timestep        
        PointScatterer scatterer;
        vec3 p(0.0f, 0.0f, 0.0f);
        cole_float amplitude = 0.0f;
        for (int i = 0; i < num_control_points; i++) {
            // TODO: implement scalar product in Vector3
            p.x += basis_functions[i]*m_scatterers->nodes[scatterer_no][i].pos.x;
            p.y += basis_functions[i]*m_scatterers->nodes[scatterer_no][i].pos.y;
            p.z += basis_functions[i]*m_scatterers->nodes[scatterer_no][i].pos.z;
            amplitude += basis_functions[i]*m_scatterers->nodes[scatterer_no][i].amplitude;
        }
        scatterer.pos = p;
        scatterer.amplitude = amplitude;
        
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
