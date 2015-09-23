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

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <stdexcept>
#include "BCSimConvenience.hpp"
#include "bspline.hpp"

namespace bcsim {

std::vector<std::vector<bc_float> > decimate_frame(const std::vector<std::vector<bc_float> >& frame, int rad_decimation) {
    if (rad_decimation < 1) throw std::runtime_error("Invalid decimation value");

    auto num_beams = frame.size();
    auto num_samples = frame[0].size();
    auto num_samples_dec = num_samples / rad_decimation;
    
    std::vector<std::vector<bc_float> > decimated_frame(num_beams);
    for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
        decimated_frame[beam_no].resize(num_samples_dec);
        for (size_t sample_no = 0; sample_no < num_samples_dec; sample_no++) {
            decimated_frame[beam_no][sample_no] = frame[beam_no][sample_no*rad_decimation];
        }
    }
    return decimated_frame;
}

bc_float get_max_value(const std::vector<std::vector<bc_float> >& image_lines) {
    std::vector<bc_float> max_values;
    for (const auto& image_line : image_lines) {
        max_values.push_back(*std::max_element(image_line.begin(), image_line.end()));
    }
    return *std::max_element(max_values.begin(), max_values.end());
}

void log_compress_frame(std::vector<std::vector<bc_float> >& image_lines, float dyn_range, float normalize_factor) {

    auto num_beams   = image_lines.size();
    auto num_samples = image_lines[0].size(); 

    for (auto& beam : image_lines) {
        std::transform(beam.begin(), beam.end(), beam.begin(), [=](bc_float pixel) {
            // log-compression
            pixel = static_cast<bc_float>(20.0*std::log10(pixel/normalize_factor));
            pixel = (255.0/dyn_range)*(pixel + dyn_range);
            
            // clamp to [0, 255]
            if (pixel < 0.0f) pixel = 0.0f;
            if (pixel >= 255.0f) pixel = 255.0f;
            
            return pixel;
        });
    }
}

Scatterers::s_ptr render_fixed_scatterers(SplineScatterers::s_ptr spline_scatterers, float timestamp) {
    // TODO: can parts of this code be put in a separate function and used both
    // here and in the CPU spline algoritm to reduce code duplication?
    auto res = FixedScatterers::s_ptr(new FixedScatterers);
    const auto num_scatterers = spline_scatterers->num_scatterers();
    if (num_scatterers == 0) {
        throw std::runtime_error("No spline scatterers");
    }
    
        
    // precompute basis functions
    const auto num_cs = spline_scatterers->nodes[0].size();
    std::vector<float> basis_fn(num_cs);
    for (size_t i = 0; i < num_cs; i++) {
        basis_fn[i] = bspline_storve::bsplineBasis(i, spline_scatterers->spline_degree, timestamp, spline_scatterers->knot_vector);
    }
    
    // evaluate using cached basis functions
    res->scatterers.resize(num_scatterers);
    for (size_t spline_no = 0; spline_no < num_scatterers; spline_no++) {
        PointScatterer scatterer;
        scatterer.pos       = vector3(0.0f, 0.0f, 0.0f);
        scatterer.amplitude = 0.0f;
        for (size_t i = 0; i < num_cs; i++) {
            scatterer.pos       += spline_scatterers->nodes[spline_no][i].pos*basis_fn[i];
            scatterer.amplitude += spline_scatterers->nodes[spline_no][i].amplitude*basis_fn[i];
        }
        res->scatterers[spline_no] = scatterer;
    }

    return res;
}


}   // end namespace
