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

#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>

template <typename T>
std::vector<std::complex<T> > fft(const std::vector<std::complex<T> >& x) {
    auto n = x.size();
    auto n_half = n/2;
    if (n == 1) {
        return x;
    }
    const auto PI = 4.0*std::atan(1.0);
    auto wn = std::exp(std::complex<T>(static_cast<T>(0.0), static_cast<T>(-2.0*PI/n)));
    auto w = std::complex<T>(static_cast<T>(1.0), static_cast<T>(0.0));
    std::vector<std::complex<T> > x_even(n_half);
    std::vector<std::complex<T> > x_odd(n_half);
    for (size_t i = 0; i < n_half; i++) {
        x_even[i] = x[2*i];
        x_odd[i]  = x[2*i+1];    
    }
    auto y_even = fft(x_even);
    auto y_odd  = fft(x_odd);
    std::vector<std::complex<T> > y(n, std::complex<T>(static_cast<T>(0.0), static_cast<T>(0.0)));
    for (size_t k = 0; k < n_half; k++) {
        y[k]        = y_even[k] + w*y_odd[k];
        y[k+n_half] = y_even[k] - w*y_odd[k];
        w = w*wn;
    }
    return y;
}

template <typename T>
std::vector<std::complex<T> > ifft(const std::vector<std::complex<T> >& x) {
    const auto n = x.size();
    // swap real and imag
    std::vector<std::complex<T> > temp(n);
    std::transform(x.begin(), x.end(), temp.begin(), [](std::complex<T> sample) {
        return std::complex<T>(sample.imag(), sample.real()); 
    });
    // take the forward FFT
    auto res = fft(temp);
    
    // swap real and imag and scale.
    std::transform(res.begin(), res.end(), temp.begin(), [=](std::complex<T> sample) {
        return std::complex<T>(sample.imag()/n, sample.real()/n); 
    });
    return temp;
}

size_t next_power_of_two(size_t n) {
    return static_cast<size_t>(std::pow(2, std::ceil(std::log(n) / std::log(2))));
}

template <typename T>
std::vector<std::complex<T> > zero_pad_to_complex(const std::vector<T>& v, size_t padded_size) {
    std::vector<std::complex<T> > res(padded_size, std::complex<T>(0.0, 0.0));
    for (size_t i = 0; i < v.size(); i++) {
        res[i] = std::complex<T>(v[i], 0.0);
    }
    return res;
}

template <typename T>
std::vector<T> fft_conv(const std::vector<T>& x, const std::vector<T>& y) {
    // TODO: Implementation can be simplified since the input is real.

    const auto final_out_size = x.size() + y.size() - 1;

    // find the power of two we must use
    const auto power_size = next_power_of_two(final_out_size);

    const auto padded_x_fft = fft(zero_pad_to_complex(x, power_size));
    const auto padded_y_fft = fft(zero_pad_to_complex(y, power_size));

    // compute product of FFTs
    std::vector<std::complex<T> > fft_prod;
    fft_prod.reserve(power_size);
    for (size_t i = 0; i < power_size; i++) {
        fft_prod.push_back(padded_x_fft[i]*padded_y_fft[i]);
    }

    // inverse FFT of product
    const auto temp_res = ifft(fft_prod);

    // extract real part
    std::vector<T> res(final_out_size);
    for (size_t i = 0; i < final_out_size; i++) {
        res[i] = temp_res[i].real();
    }
    return res;
}


// explicit instantiations for float and double.
template std::vector<std::complex<float> >  fft(const std::vector<std::complex<float> >& x);
template std::vector<std::complex<double> > fft(const std::vector<std::complex<double> >& x);
template std::vector<std::complex<float> >  ifft(const std::vector<std::complex<float> >& x);
template std::vector<std::complex<double> > ifft(const std::vector<std::complex<double> >& x);
template std::vector<float>  fft_conv(const std::vector<float>& x, const std::vector<float>& y);
template std::vector<double> fft_conv(const std::vector<double>& x, const std::vector<double>& y);
