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
#include <memory>
#include "export_macros.hpp"
#include "bcsim_defines.h"
#include "BCSimConfig.hpp"
#include "ScanGeometry.hpp"

namespace bcsim {

// A single scanline.
class DLL_PUBLIC Scanline {
public:
    Scanline() { }
    
    // Elevational direction implicitly defined as cross product between lateral_dir and direction.
    Scanline(const vector3& origin, const vector3& direction, const vector3& lateral_dir, float timestamp);
    
    vector3 get_elevational_dir() const {
        return elevational_dir;
    }

    vector3 get_direction() const {
        return direction;
    }

    vector3 get_origin() const {
        return origin;
    }

    vector3 get_lateral_dir() const {
        return lateral_dir;
    }

    float get_timestamp() const {
        return timestamp;
    }

    // Verify that the unit vectors are orthonormal
    bool is_valid() const;
        
private:
    // The start point.
    vector3 origin;

    // The direction unit vector (radial direction).
    vector3 direction;
    
    // Unit vector in lateral direction (the elevational direction
    // is implicitly defined by the cross product of lateral_dir and direction)
    vector3 lateral_dir;
    
    // Timestamp for scanline which is useful with dynamic scatterer collections.
    float timestamp;

    // Computed as cross product between the lateral and radial direction.
    vector3 elevational_dir;

private:
    // Verify that the unit vectors span 3D space.
    bool is_orthogonal() const;

    // Verify that the vectors have unit length.
    bool is_normalized() const;
};

// A collection of scanlines that defines the complete scan.
// The length of scanlines is common to all lines in a scan sequence.
class DLL_PUBLIC ScanSequence {
public:
    typedef std::shared_ptr<ScanSequence> s_ptr;
    typedef std::unique_ptr<ScanSequence> u_ptr;

    ScanSequence(float line_length);
    
    int get_num_lines() const;
    
    // Add a new scanline to the scan sequence.
    void add_scanline(Scanline new_line);
    
    const Scanline& get_scanline(int index) const;

    // Verify that all scanlines are valid.
    bool is_valid() const;

    // The beam length [m]
    float line_length;

private:
    std::vector<Scanline> scanlines;
};

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

}   // namespace

