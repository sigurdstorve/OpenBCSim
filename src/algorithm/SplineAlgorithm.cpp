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
#include "bcsim_defines.h"
#include "SplineAlgorithm.hpp"
#include "bspline.hpp"
#include "safe_omp.h"

namespace bcsim {

SplineAlgorithm::SplineAlgorithm()
        : CpuBaseAlgorithm() { }
     
void SplineAlgorithm::set_scatterers(Scatterers::s_ptr new_scatterers) {
    m_scatterers = std::dynamic_pointer_cast<SplineScatterers>(new_scatterers);

    if (m_scatterers == nullptr) {
        throw std::runtime_error("Cast to SplineScatterers failed");
    }

    if (m_verbose) std::cout << "Number of spline scatterers: " << m_scatterers->num_scatterers() << std::endl;
    m_scatterers_configured = true;
}

void SplineAlgorithm::projection_loop(const Scanline& line, double* time_proj_signal, size_t num_time_samples) {

    const int num_scatterers = m_scatterers->nodes.size();
    // TODO: Improve. Use that all splines have same number of control points
    const int num_control_points = m_scatterers->nodes[0].size();
    std::vector<bc_float> basis_functions(num_control_points);
    
    // Precompute all B-spline basis function for current timestep
    for (int i = 0; i < num_control_points; i++) {
        const bc_float b = bspline_storve::bsplineBasis(i,
                                                        m_scatterers->spline_degree,
                                                        line.get_timestamp(),
                                                        m_scatterers->knot_vector);
        basis_functions[i] = b;
    }
    for (int scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
        // Compute position of current scatterer by evaluating spline in current timestep        
        PointScatterer scatterer;
        scatterer.pos = vector3(0.0f, 0.0f, 0.0f); // TODO: not neccessary to set to zero since constructor does that?
        scatterer.amplitude = 0.0f;
        for (int i = 0; i < num_control_points; i++) {
            scatterer.pos       += m_scatterers->nodes[scatterer_no][i].pos*basis_functions[i];
            scatterer.amplitude += m_scatterers->nodes[scatterer_no][i].amplitude*basis_functions[i];
        }
        
        // Map the global cartesian scatterer position into the beam's local
        // coordinate system.
        vector3 temp = scatterer.pos - line.get_origin();
        bc_float r = temp.dot(line.get_direction());       // radial component
        bc_float l = temp.dot(line.get_lateral_dir());     // lateral component
        bc_float e = temp.dot(line.get_elevational_dir()); // elevational component
        
        // Add scaled amplitude to closest index
        const bc_float sampling_time_step = 1.0/m_excitation.sampling_frequency;
        int closest_index = (int) std::floor(r*2.0/(m_sound_speed*sampling_time_step)+0.5f);
        
        bc_float scaled_ampl = m_beamProfile->sampleProfile(r,l,e)*scatterer.amplitude;
        
        // Avoid out of bound seg.fault
        if (closest_index < 0 || closest_index >= num_time_samples) {
#if BCSIM_PRINT_SCATTERER_OUTSIDE_MSG
            std::cout << "[warning] scatterer outside scan line." << std::endl;
#endif
            continue;
        }
        time_proj_signal[closest_index] += scaled_ampl;
    }
}

}   // namespace
