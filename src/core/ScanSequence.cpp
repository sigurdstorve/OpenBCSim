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
#include "ScanSequence.hpp"
#include "vector3.hpp"

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

}   // namespace
