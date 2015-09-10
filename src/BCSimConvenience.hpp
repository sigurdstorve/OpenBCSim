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
#include <vector>
#include "export_macros.hpp"
#include "BCSimConfig.hpp"

// misc. utilities for post processing of simulated data
namespace bcsim {

// Decimate all beams in a frame radially.
// frame:               vector of beams
// radial_decimation:   radial decimation factor (>= 1)
std::vector<std::vector<bc_float> > DLL_PUBLIC decimate_frame(const std::vector<std::vector<bc_float> >& frame, int rad_decimation);

// Determine the biggest value in all beams in frame [typ. for envelope detected data]
bc_float DLL_PUBLIC get_max_value(const std::vector<std::vector<bc_float> >& env_frame);

// Log-compress every pixel [assuming that the input is envelope detected]
// dyn_range:           The dynamic range [dB]
// normalize_factor:    Normalization factor [typ. determined as max over all beams in all frames]
void DLL_PUBLIC log_compress_frame(std::vector<std::vector<bc_float> >& env_frame, float dyn_range, float normalize_factor);

}   // end namespace