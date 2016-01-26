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
#include "ScanSequence.hpp"
#include "ScanGeometry.hpp"

// Misc. utilities for post processing of simulated data and for creating
// scan sequences.
namespace bcsim {

// Decimate all beams in a frame radially.
// frame:               vector of beams
// radial_decimation:   radial decimation factor (>= 1)
std::vector<std::vector<float> > DLL_PUBLIC decimate_frame(const std::vector<std::vector<float> >& frame, int rad_decimation);

// Determine the biggest value in all beams in frame [typ. for envelope detected data]
float DLL_PUBLIC get_max_value(const std::vector<std::vector<float> >& image_lines);

// Log-compress every pixel and clamp result to [0, 255]
// dyn_range:           The dynamic range [dB]
// normalize_factor:    Normalization factor [typ. determined as max over all beams in all frames]
// gain_factor:         Image gain
void DLL_PUBLIC log_compress_frame(std::vector<std::vector<float> >& image_lines, float dyn_range, float normalize_factor, float gain_factor);

// Evaluate a spline scatterer dataset at a specific time in order to generate
// a new fixed scatterer dataset.
// timestamp: the time to evaluate the spline scatterers in
// NOTE: This code is completely untested!
Scatterers::s_ptr DLL_PUBLIC render_fixed_scatterers(SplineScatterers::s_ptr spline_scatterers, float timestamp);

// Create a sector/linear scan where all lines have the same timestamp.
// By convention, all scan sequences are created in their own local coordinate system
// centered at origin. The standard radial direction is along the z-axis and the lateral
// direction is along the x-axis.
ScanSequence DLL_PUBLIC CreateScanSequence(ScanGeometry::ptr geometry, size_t num_lines, float timestamp);

// Orient the lines in a ScanSequence by
// 1. Rotate using x,y, and z rotation angles, in that order, i.e.
//      p_rot = R*p where R = R_z*R_y*R_x
// 2. Translate the origin.
ScanSequence::s_ptr DLL_PUBLIC OrientScanSequence(const ScanSequence& scan_seq, const vector3& rot_angles, const vector3& origin);

}   // end namespace