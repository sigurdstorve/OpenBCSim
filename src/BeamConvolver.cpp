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
#include <complex>
#include <algorithm>
#include <functional>
#include "discrete_hilbert_mask.hpp"
#include "fft.hpp"
#include "BeamConvolver.hpp"

namespace bcsim {

// Common functionality for all the types of convolvers
class BeamConvolverBase : public IBeamConvolver {
public:
    // num_proj_samples: Number of time-projection samples
    // excitation: The RF excitation
    // out_transform: The mapping from complex to real. Default is to extract the real part.
    BeamConvolverBase(size_t num_proj_samples, const ExcitationSignal& excitation, std::function<bc_float(std::complex<float>)> out_transform)
        : m_out_transform(out_transform),
          m_excitation_has_been_customized(false)
    {
        m_num_conv_samples = num_proj_samples + excitation.samples.size() - 1;
        m_time_proj_buffer.resize(num_proj_samples);
        m_fft_length = next_power_of_two(m_num_conv_samples);
        precompute_excitation_fft(excitation);
        m_excitation_delay = static_cast<size_t>(excitation.center_index);
        // customization happens later since virtual functions cannot be called here... 

        m_temp_buffer.resize(m_fft_length);
    }

    virtual double* get_zeroed_time_proj_signal() {
        std::fill(m_time_proj_buffer.begin(), m_time_proj_buffer.end(), 0.0);
        return m_time_proj_buffer.data();
    }

    // Use contents of the time-projected buffer and create an RF line.
    virtual std::vector<bc_float> process() {
        process_first_stage();

        // extract output, compensate for delay introduced by convolving with excitation
        auto num_proj_samples = m_time_proj_buffer.size();
        auto start = m_temp_buffer.begin() + m_excitation_delay;
        std::vector<bc_float> res(num_proj_samples);
        std::transform(start, start + num_proj_samples, res.begin(), [&](std::complex<float> v) {
            return m_out_transform(v);
        });
        return res;
    }

protected:
    // Customization step to alter the precomputed FFT of excitation signal, which
    // is useful for implementing an additional Hilbert transform at no extra cost.
    virtual void customize_excitation_fft() {
        // default is to do nothing.
    }

    void precompute_excitation_fft(const ExcitationSignal& excitation) {
        std::vector<std::complex<float>> padded_excitation(m_fft_length, std::complex<float>(0.0f, 0.0f));
        std::transform(excitation.samples.begin(), excitation.samples.end(), padded_excitation.begin(), [](bc_float v) {
            return std::complex<float>(static_cast<float>(v), 0.0f);
        });
        m_excitation_fft = fft(padded_excitation);
    }

    // Process the time-projections by doing FFT -> Multiply -> IFFT. Virtual to enable bypassing it.
    // Result is stored in m_temp_buffer
    virtual void process_first_stage() {
        if (!m_excitation_has_been_customized) {
            customize_excitation_fft();
            m_excitation_has_been_customized = true;
        }
        std::vector<std::complex<float> > padded_input(m_fft_length, std::complex<float>(0.0f, 0.0f));
        std::transform(m_time_proj_buffer.begin(), m_time_proj_buffer.end(), padded_input.begin(), [](double value) {
            return std::complex<float>(static_cast<float>(value), 0.0f);
        });
        m_temp_buffer = fft(padded_input);
        if (m_temp_buffer.size() != m_fft_length) throw std::logic_error("should not happen");
        if (m_excitation_fft.size() != m_fft_length) throw std::logic_error("should not happen");
        std::transform(m_temp_buffer.begin(), m_temp_buffer.end(), m_excitation_fft.begin(), m_temp_buffer.begin(), std::multiplies<std::complex<float>>());
        m_temp_buffer = ifft(m_temp_buffer);
        if (m_temp_buffer.size() != m_fft_length) throw std::logic_error("should not happen");
    }

protected:
    size_t                              m_num_conv_samples;   // length of convolution output  [len(time_proj)+len(excitation)-1 samples]      
    std::vector<double>                 m_time_proj_buffer;   // where time-projections are stored in projection loop
    size_t                              m_fft_length;         // closest power-of-two >= length(m_time_proj_buffer)
    std::vector<std::complex<float>>    m_excitation_fft;     // Forward FFT of padded excitation, length is m_fft_length
    bool                                m_excitation_has_been_customized;
    size_t                              m_excitation_delay;   // Compensation offset needed since time zero in the middle.
    std::vector<std::complex<float>>    m_temp_buffer;        // buffer for holding intermediate results, length is m_fft_length
    std::function<bc_float(std::complex<float>)> m_out_transform;
};

// Beam convolver which returns RF samples
class BeamConvolver_RF : public BeamConvolverBase {
public:
    BeamConvolver_RF(size_t num_proj_samples, const ExcitationSignal& excitation)
        : BeamConvolverBase(num_proj_samples, excitation, [](std::complex<float> v) { return v.real(); })
    { }
};

// Beam convolver which return envelope-detected data.
class BeamConvolver_Env : public BeamConvolverBase {
public:
    BeamConvolver_Env(size_t num_proj_samples, const ExcitationSignal& excitation)
    : BeamConvolverBase(num_proj_samples, excitation, [](std::complex<float> v) { return std::abs(v); })
    { }

protected:
    virtual void customize_excitation_fft() override {
        auto h = discrete_hilbert_mask<float>(m_fft_length);
        std::transform(h.begin(), h.end(), m_excitation_fft.begin(), m_excitation_fft.begin(), [](float mask, std::complex<float> v) {
            return v*mask;
        });
    }
};

// Beam convolver which returns the time-projections
class BeamConvolver_TimeProj : public BeamConvolverBase {
public:
    BeamConvolver_TimeProj(size_t num_proj_samples, const ExcitationSignal& excitation)
        : BeamConvolverBase(num_proj_samples, excitation, [](std::complex<float> v) { return v.real(); })
    { }

protected:
    virtual void process_first_stage() override {
        std::transform(m_time_proj_buffer.begin(), m_time_proj_buffer.end(), m_temp_buffer.begin(), [](float v) {
            return std::complex<float>(v, 0.0f);
        });
    }
};

IBeamConvolver::ptr IBeamConvolver::Create(const std::string& type, size_t num_proj_samples, const ExcitationSignal& excitation) {
    IBeamConvolver* temp = nullptr;
    if (type == "rf") {
        temp = new BeamConvolver_RF(num_proj_samples, excitation);
    } else if (type == "env") {
        temp = new BeamConvolver_Env(num_proj_samples, excitation);
    } else if (type == "proj") {
        temp = new BeamConvolver_TimeProj(num_proj_samples, excitation);
    } else {
        throw std::runtime_error("Unknown beam convolver type: " + type);
    }
    return IBeamConvolver::ptr(temp);
}

}   // end namespace
