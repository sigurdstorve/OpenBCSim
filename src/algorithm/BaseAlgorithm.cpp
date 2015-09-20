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
#include "BeamConvolver.hpp"


namespace bcsim {
    
std::string to_string(const OutputType& output_type) {
    switch(output_type) {
    case OutputType::RF_DATA:
        return "rf";
        break;
    case OutputType::ENVELOPE_DATA:
        return "env";
        break;
    case OutputType::PROJECTIONS:
        return "proj";
        break;
    default:
        throw std::logic_error("invalid output type");
    }
}

IBeamConvolver::ptr CreateBeamConvolver(const OutputType& output_type,
                                        size_t num_proj_samples,
                                        const ExcitationSignal& excitation) {
    return IBeamConvolver::Create(to_string(output_type), num_proj_samples, excitation);
}    
    
    
BaseAlgorithm::BaseAlgorithm()
    : m_param_verbose(0),
      m_param_output_type(OutputType::RF_DATA),
      m_param_sound_speed(1540.0f),
      m_param_noise_amplitude(0.0f)
{
}

void BaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "verbose") {
        const auto verbose = std::stoi(value);
        m_param_verbose = verbose;
    } else if (key == "output_type") {
        if (value == "rf") {
            m_param_output_type = OutputType::RF_DATA;
        } else if (value == "env") {
            m_param_output_type = OutputType::ENVELOPE_DATA;
        } else if (value == "proj") {
            m_param_output_type = OutputType::PROJECTIONS;            
        } else {
            throw std::runtime_error(std::string("Illegal output_type '") + value + std::string("'"));
        }
    } else if (key == "sound_speed") {
        const auto new_speed = std::stof(value);
        if (new_speed <= 0) {
            throw std::runtime_error("illegal sound speed");
        }
        m_param_sound_speed = new_speed;
    } else if (key == "noise_amplitude") {
        const auto new_amplitude = std::stof(value);
        m_param_noise_amplitude = new_amplitude;
    } else {
        const auto err_msg = std::string("illegal parameter name: '") + key + std::string("'");
        throw std::runtime_error(err_msg);
    }
}

}   // end namespace

