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
#include "BaseAlgorithm.hpp"
#include "../BeamConvolver.hpp"


namespace bcsim {
    
BaseAlgorithm::BaseAlgorithm()
    : m_param_verbose(0),
      m_param_sound_speed(1540.0f),
      m_param_noise_amplitude(0.0f),
      m_param_use_arc_projection(true),
      m_radial_decimation(1),
      m_enable_phase_delay(false),
      m_cur_beam_profile_type(BeamProfileType::NOT_CONFIGURED)
{
}

void BaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "verbose") {
        const auto verbose = std::stoi(value);
        m_param_verbose = verbose;
    } else if (key == "sound_speed") {
        const auto new_speed = std::stof(value);
        if (new_speed <= 0) {
            throw std::runtime_error("illegal sound speed");
        }
        m_param_sound_speed = new_speed;
    } else if (key == "noise_amplitude") {
        const auto new_amplitude = std::stof(value);
        m_param_noise_amplitude = new_amplitude;
    } else if (key == "use_arc_projection") {
        if (value == "on" || value == "true") {
            m_param_use_arc_projection = true;
        } else if (value == "off" || value == "false") {
            m_param_use_arc_projection = false;
        } else {
            throw std::runtime_error("invalid boolean value");
        }
    } else if (key == "radial_decimation") {
        const auto new_radial_decimation = std::stoi(value);
        if (new_radial_decimation <= 0) {
            throw std::runtime_error("illegal radial decimation value");
        }
        m_radial_decimation = new_radial_decimation;
    } else if (key == "phase_delay") {
        if (value == "on" || value == "true") {
            m_enable_phase_delay = true;
        } else if (value == "off" || value == "false") {
            m_enable_phase_delay = false;
        } else { 
            throw std::runtime_error("invalid boolean value");
        }
    } else {
        const auto err_msg = std::string("illegal parameter name: '") + key + std::string("'");
        throw std::runtime_error(err_msg);
    }
}

std::vector<double> BaseAlgorithm::get_debug_data(const std::string& identifier) const {
    if (m_debug_data.find(identifier) == std::end(m_debug_data)) {
        throw std::runtime_error("invalid debug data key: " + identifier);
    }
    return m_debug_data.at(identifier);
}

std::string BaseAlgorithm::get_parameter(const std::string& key) const {
    throw std::runtime_error("Illegal key: " + key);
}

}   // end namespace

