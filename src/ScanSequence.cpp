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

#include <stdexcept>
#include <cmath>
#include "bcsim_defines.h"
#include "ScanSequence.hpp"
#include "vector3.hpp"
#include "rotation3d.hpp"

namespace bcsim {

Scanline::Scanline(const vector3& origin, const vector3& direction, const vector3& lateral_dir, float timestamp) :
    origin(origin), direction(direction), lateral_dir(lateral_dir), timestamp(timestamp) {

    elevational_dir = lateral_dir.cross(direction); 

    // TODO: Should all vectors be normalized here?

    if (!is_orthogonal()) throw std::runtime_error("Invalid Scanline: Not orthogonal unit vectors");
    if (!is_normalized()) throw std::runtime_error("Invalid Scanline: Not unit length vectors");
}

bool Scanline::is_valid() const {
    return (is_orthogonal() && is_normalized());
}

bool Scanline::is_orthogonal() const {
    const float THRESHOLD = 1e-4f;
    bool res = true;
    auto elevational_dir = get_elevational_dir();
    if (std::abs(direction.dot(lateral_dir))       >= THRESHOLD) res = false;
    if (std::abs(lateral_dir.dot(elevational_dir)) >= THRESHOLD) res = false;
    if (std::abs(elevational_dir.dot(direction))   >= THRESHOLD) res = false;
    return res;
}

bool Scanline::is_normalized() const {
    auto elevational_dir = get_elevational_dir();
    const float THRESHOLD = 1e-4f;
    bool res = true;
    if (std::abs(direction.norm() - 1.0)       >= THRESHOLD) res = false;
    if (std::abs(lateral_dir.norm() - 1.0)     >= THRESHOLD) res = false;
    if (std::abs(elevational_dir.norm() - 1.0) >= THRESHOLD) res = false;
    return res;
}

ScanSequence::ScanSequence(float line_length)
    : line_length(line_length)
{
}

int ScanSequence::get_num_lines() const {
    return static_cast<int>(scanlines.size());
}

void ScanSequence::add_scanline(Scanline new_line) {
    scanlines.push_back(new_line);
}

const Scanline& ScanSequence::get_scanline(int index) const {
    if (index < 0 || index >= get_num_lines()) {
        throw std::runtime_error("ScanSequence::Invalid index.");
    }
    return scanlines[index];
}

bool ScanSequence::is_valid() const {
    bool res = true;
    int num_lines = static_cast<int>(get_num_lines());
    for (int line_no = 0; line_no < num_lines; line_no++) {
        if (!get_scanline(line_no).is_valid()) res = false;
    }
    return res;
}

ScanSequence CreateScanSequence(std::shared_ptr<SectorScanGeometry> geometry, size_t num_lines, float timestamp) {
    const auto line_length = geometry->depth;
    ScanSequence res(line_length);
    
    // Will be transformed by a rotation into lateral and radial unit vectors.
    // NOTE: Need to use double to avoid "not orthonormal" error (maybe the test is to strict for floats?)
    const auto unit_vector_x = unit_x<double>();
    const auto unit_vector_z = unit_z<double>();
    const vector3 origin(0.0f, 0.0f, 0.0f);

    for (int line_no = 0; line_no < static_cast<int>(num_lines); line_no++) {
        const float angle = -0.5f*geometry->width + geometry->tilt + line_no*geometry->width/(num_lines-1);

        const auto ROT_MATRIX = rotation_matrix_y<double>(angle);
        const auto temp_radial_direction  = boost::numeric::ublas::prod(ROT_MATRIX, unit_vector_z);
        const auto temp_lateral_direction = boost::numeric::ublas::prod(ROT_MATRIX, unit_vector_x); 
	
        // Copy to vector3 vectors. TODO: deduplicate
        const vector3 direction  ((float)temp_radial_direction(0),  (float)temp_radial_direction(1),  (float)temp_radial_direction(2));
        const vector3 lateral_dir((float)temp_lateral_direction(0), (float)temp_lateral_direction(1), (float)temp_lateral_direction(2));

        try {
            auto sl = Scanline(origin, direction, lateral_dir, timestamp);
            res.add_scanline(sl);
        } catch (std::runtime_error& e) {
            throw std::runtime_error(std::string("failed creating scan line: ") + e.what());
        }
        
    }
    return res;
}

ScanSequence CreateScanSequence(std::shared_ptr<LinearScanGeometry> geometry, size_t num_lines, float timestamp) {
    const auto line_length = geometry->range_max;
    ScanSequence res(line_length);

    const auto unit_vector_x = unit_x<double>();
    const auto unit_vector_z = unit_z<double>();

    // Copy to vector3 vectors. TODO: deduplicate
    const vector3 direction  (unit_vector_z(0), unit_vector_z(1), unit_vector_z(2));
    const vector3 lateral_dir(unit_vector_x(0), unit_vector_x(1), unit_vector_x(2));


    for (int line_no = 0; line_no < static_cast<int>(num_lines); line_no++) {
        try {
            const vector3 scanline_origin(-0.5f*geometry->width + line_no*geometry->width/(static_cast<int>(num_lines)-1), 0.0f, 0.0f);
            auto sl = Scanline(scanline_origin, direction, lateral_dir, timestamp);

            res.add_scanline(sl);
        } catch (std::runtime_error& e) {
            throw std::runtime_error(std::string("failed creating scan line: ") + e.what());
        }
    }
    
    return res;
}

// probe_origin is position of probe's origin in world coordinate system.
ScanSequence CreateScanSequence(ScanGeometry::ptr geometry, size_t num_lines, float timestamp) {
    auto sector_geo = std::dynamic_pointer_cast<SectorScanGeometry>(geometry); 
    auto linear_geo = std::dynamic_pointer_cast<LinearScanGeometry>(geometry);
    if (sector_geo) {
        return CreateScanSequence(sector_geo, num_lines, timestamp);
    } else if (linear_geo) {
        return CreateScanSequence(linear_geo, num_lines, timestamp);
    } else {
        throw std::runtime_error("unable to cast scan geometry");
    }
}

namespace detail {
    // Apply a 3x3 rotation matrix to a vector3;
    template <typename T>
    vector3 TransformVector(const vector3& v, const boost::numeric::ublas::matrix<T>& matrix33) {
        boost::numeric::ublas::vector<T> temp_v(3);
        temp_v(0) = static_cast<T>(v.x); temp_v(1) = static_cast<T>(v.y); temp_v(2) = static_cast<T>(v.z);
        const auto transformed = boost::numeric::ublas::prod(matrix33, temp_v);
        return vector3(static_cast<bc_float>(transformed(0)),
                       static_cast<bc_float>(transformed(1)),
                       static_cast<bc_float>(transformed(2)));
    }
}

ScanSequence::s_ptr OrientScanSequence(const ScanSequence& scan_seq, const vector3& rot_angles, const vector3& probe_origin) {
    const auto rot_matrix = rotation_matrix_xyz(rot_angles.x, rot_angles.y, rot_angles.z);

    const auto line_length = scan_seq.line_length;

    auto num_lines = scan_seq.get_num_lines();
    auto res = new ScanSequence(line_length);
    for (int i = 0; i < num_lines; i++) {
        const auto old_line = scan_seq.get_scanline(i);
        const auto rotated_origin      = detail::TransformVector(old_line.get_origin(),      rot_matrix);
        const auto rotated_direction   = detail::TransformVector(old_line.get_direction(),   rot_matrix);
        const auto rotated_lateral_dir = detail::TransformVector(old_line.get_lateral_dir(), rot_matrix);
        
        res->add_scanline(Scanline(rotated_origin + probe_origin,
                                  rotated_direction,
                                  rotated_lateral_dir,
                                  old_line.get_timestamp()));
    }
    return ScanSequence::s_ptr(res);
}


}   // namespace

