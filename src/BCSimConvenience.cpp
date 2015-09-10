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
#include <cmath>
#include <stdexcept>
#include "BCSimConvenience.hpp"

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

bc_float get_max_value(const std::vector<std::vector<bc_float> >& env_frame) {
    std::vector<bc_float> max_values;
    for (const auto& env_line : env_frame) {
        max_values.push_back(*std::max_element(env_line.begin(), env_line.end()));
    }
    return *std::max_element(max_values.begin(), max_values.end());
}

void log_compress_frame(std::vector<std::vector<bc_float> >& env_frame, float dyn_range, float normalize_factor) {

    auto num_beams   = env_frame.size();
    auto num_samples = env_frame[0].size(); 

    for (auto& beam : env_frame) {
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


}   // end namespace
