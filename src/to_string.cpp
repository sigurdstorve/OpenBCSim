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

#include <sstream>
#include "to_string.hpp"

namespace bcsim {

std::string to_string(const ExcitationSignal& sig) {
    std::stringstream ss;
    ss << "--ExcitationSignal--" << std::endl;
    ss << "Center index: " << sig.center_index << std::endl;
    ss << "Sampling frequency: " << sig.sampling_frequency << std::endl;
    ss << "Demodulation frequency: " << sig.demod_freq << std::endl;
    return ss.str();
}

std::string to_string(const PointScatterer& ps) {
    std::stringstream ss;
    ss << "@" << to_string(ps.pos) << ",A=" << ps.amplitude;
    return ss.str();
}

std::string to_string(const Scanline& line) {
    std::stringstream ss;
    ss << "Origin: " << to_string(line.get_origin()) 
       << " Direction: " << to_string(line.get_direction()) << " Lateral dir: " << to_string(line.get_lateral_dir())
       << " Elevational dir: " << to_string(line.get_elevational_dir())
       << std::endl;;
    return ss.str();
}

std::string to_string(const ScanSequence& seq) {
    std::stringstream ss;
    auto num_lines = seq.get_num_lines();
    ss << "ScanSequence with" << num_lines << " with common length " << seq.line_length << std::endl;
    for (int i=0; i < num_lines; i++) {
        ss << to_string(seq.get_scanline(i)) << std::endl;
    }
    return ss.str();
}

std::string to_string(const vector3& v) {
    std::stringstream ss;
    ss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return ss.str();
}


}   // namespace

