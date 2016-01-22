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

#pragma once
#include <cmath>

namespace bcsim {

// Compute the required number of RF samples for a single image line
// given the sound speed [m/s], line length [m], and temporal sampling freq. [Hz]
template <typename T>
size_t compute_num_rf_samples(T sound_speed, T line_length, T sampling_freq) {
    auto max_time = static_cast<T>(2.0*line_length/sound_speed);
    return static_cast<size_t>(std::floor(sampling_freq*max_time + 0.5)); 
}

// When evaluating a spline as a sum of control points and basis functions,
// only degree+1 terms are non-zero. The start and end index (inclusive) can
// be computed. This function asserts that the skipped basis functions are in
// fact zero. Note that this function relies on a vector of all evaluated basis
// functions.
//
// Returns true if test passes.
template <typename T>
bool sanity_check_spline_lower_upper_bound(const std::vector<T>& all_basis_functions,
                                           int cs_idx_start, int cs_idx_end) {
    for (int i = 0; i < cs_idx_start; i++) {
        if (std::abs(all_basis_functions[i]) > 1e-6) return false;
    }
    for (int i = cs_idx_end+1; i < all_basis_functions.size(); i++) {
        if (std::abs(all_basis_functions[i]) > 1e-6) return false;
    }
    return true;
}

}   // end namespace
