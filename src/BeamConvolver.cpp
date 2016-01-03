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
    BeamConvolverBase(size_t num_proj_samples, const ExcitationSignal& excitation)
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
    // Process the time-projections by doing FFT -> Multiply -> IFFT
    virtual std::vector<std::complex<bc_float>> process() {
        // Zero-pad and extend time-prohjection signal to complex domain
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

        // extract output, compensate for delay introduced by convolving with excitation
        auto num_proj_samples = m_time_proj_buffer.size();
        auto start = m_temp_buffer.begin() + m_excitation_delay;
        return std::vector<std::complex<bc_float>>(start, start + num_proj_samples);
    }

protected:
    // Precompute Hilbert-transformed FFT of excitation signal.
    void precompute_excitation_fft(const ExcitationSignal& excitation) {
        std::vector<std::complex<float>> padded_excitation(m_fft_length, std::complex<float>(0.0f, 0.0f));
        std::transform(excitation.samples.begin(), excitation.samples.end(), padded_excitation.begin(), [](bc_float v) {
            return std::complex<float>(static_cast<float>(v), 0.0f);
        });
        m_excitation_fft = fft(padded_excitation);

        // Hilbert transform is implemented by zeroing out negative frequencies in FFT of excitation.
        const auto hilbert_mask = discrete_hilbert_mask<float>(m_fft_length);
        std::transform(std::begin(hilbert_mask), std::end(hilbert_mask),
                       std::begin(m_excitation_fft), std::begin(m_excitation_fft),
                       [&](float mask_sample, std::complex<float> fft_sample) {
            return mask_sample*fft_sample;
        });
    }


protected:
    size_t                              m_num_conv_samples;   // length of convolution output  [len(time_proj)+len(excitation)-1 samples]      
    std::vector<double>                 m_time_proj_buffer;   // where time-projections are stored in projection loop
    size_t                              m_fft_length;         // closest power-of-two >= length(m_time_proj_buffer)
    std::vector<std::complex<float>>    m_excitation_fft;     // Forward FFT of padded excitation, length is m_fft_length
    size_t                              m_excitation_delay;   // Compensation offset needed since time zero in the middle.
    std::vector<std::complex<float>>    m_temp_buffer;        // buffer for holding intermediate results, length is m_fft_length
};
IBeamConvolver::ptr IBeamConvolver::Create(size_t num_proj_samples, const ExcitationSignal& excitation) {
    return IBeamConvolver::ptr(new BeamConvolverBase(num_proj_samples, excitation));
}

}   // end namespace
