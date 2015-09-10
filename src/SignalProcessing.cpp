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

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "SignalProcessing.hpp"

template <typename T>
std::vector<T> HammingWindow(size_t length) {
    std::vector<T> samples(length);
    const T PI = static_cast<T>(4*std::atan(1));
    for (size_t n = 0; n < length; n++) {
        const T sample = static_cast<T>(0.54 - 0.46*std::cos(2.0*PI*n/(length-1)));
        samples[n] = sample;
    }
    return samples;
}

template <typename T>
std::vector<T> FirWin(int length, T fc) {
    if ((length % 2) == 0) {
        throw std::runtime_error("Length must be odd");
    }

    std::vector<T> h;
    h.reserve(length);
    const double PI = 4*std::atan(1);
    for (int n = -(length-1)/2; n <= (length-1)/2; n++) {
        T temp;
        if (n == 0) {
            temp = 2*fc;
        } else {
            temp = std::sin(2*PI*fc*n)/(PI*n);
        }
        h.push_back(temp);
    }
    
    
    // Apply a Hamming window
    auto hamming_win = HammingWindow<T>(length);
    T sum = static_cast<T>(0.0);
    for (size_t i = 0; i < length; i++) {
        h[i] *= hamming_win[i];
        sum += h[i];
    }
    
    // Divide by sum of coefficients to ensure unity DC gain.
    std::transform(h.begin(), h.end(), h.begin(), [=](T value) {
        return value / sum;
    });
    
    return h;
}

template <typename T>
std::vector<T> direct_conv(const std::vector<T>& v1, const std::vector<T>& v2) {
    int s1 = static_cast<int>(v1.size());
    int s2 = static_cast<int>(v2.size());
    // c[n] = sum_i a[i]b[n-i]
    // (1) i >= 0
    // (2) i <= s1-1
    // (3) n-i >= 0 --> i <= n
    // (4) n-i <= s2-1 --> i >= n-s2+1
    int res_size = s1+s2-1;
    std::vector<T> res(res_size);
    for (int n = 0; n < res_size; n++) {
        int i_min = std::max(0, n-s2+1);
        int i_max = std::min(s1-1, n);
        T sum = static_cast<T>(0.0);
        for (int i = i_min; i <= i_max; i++) {
            sum += v1[i]*v2[n-i];
        }
        res[n] = sum;
    }
    return res;
}

// Explicit instantiations since templates cannot be exported in DLL.
template std::vector<float>  HammingWindow<float>(size_t length);
template std::vector<double> HammingWindow<double>(size_t length);
template std::vector<float>  FirWin(int length, float fc);
template std::vector<double> FirWin(int length, double fc);
template std::vector<float>  direct_conv(const std::vector<float>& v1, const std::vector<float>& v2);
template std::vector<double> direct_conv(const std::vector<double>& v1, const std::vector<double>& v2);
template std::vector<int>    direct_conv(const std::vector<int>& v1, const std::vector<int>& v2);
