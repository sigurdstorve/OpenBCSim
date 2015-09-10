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

#include "GaussPulse.hpp"
namespace bcsim {

double GaussianPulseTimeLimits(double fc, double bw, double bwr, double tpr) {
    const double PI = 4*std::atan(1);
    const double E = std::exp(1.0);
    const double f_cutoff = bw*fc;
    return std::sqrt(-tpr/(20.0*(-PI*PI*f_cutoff*f_cutoff/(std::log( std::pow(10.0, bwr/20.0) )))*std::log10(E)));
}

template <typename T>
T GaussianPulse(T t, T fc, T bw, T bwr) {
    
    const T PI = static_cast<T>(4*std::atan(1));
    const T E = static_cast<T>(std::exp(1.0));
    const T f_cutoff = bw*fc;
        
    const T freq_time_prod = static_cast<T>(f_cutoff*t);
    return static_cast<T>( std::exp(PI*PI*freq_time_prod*freq_time_prod/(std::log( std::pow(10.0, bwr/20.0) )))*std::cos(2*PI*fc*t) );
}

template <typename T>
void MakeGaussianExcitation(T fc, T bw, T fs,
                           std::vector<T>& /*out*/ times,
                           std::vector<T>& /*out*/ samples,
                           int& /*out*/ center_index) {
    // compute the number of samples needed
    auto gauss_time_limit = bcsim::GaussianPulseTimeLimits(fc, bw);
    int num_samples = static_cast<int>(2*gauss_time_limit*fs); // times two since time is for one side only 
    
    times.resize(num_samples);
    samples.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        const float time = (i-num_samples/2)/fs;
        times[i] = time;
        samples[i] = GaussianPulse<float>(time, fc, bw, -6.0);
    }
    
    // signal is symmetric around time zero
    center_index = num_samples/2;
}

// explicit template instantiations
template float  GaussianPulse(float t, float fc, float bw, float bwr);

template double GaussianPulse(double t, double fc, double bw, double bwr);

template void   MakeGaussianExcitation(float fc, float bw, float fs,
                                       std::vector<float>& /*out*/ times,
                                       std::vector<float>& /*out*/ samples,
                                       int& /*out*/ center_index);

template void   MakeGaussianExcitation(double fc, double bw, double fs,
                                       std::vector<double>& /*out*/ times,
                                       std::vector<double>& /*out*/ samples,
                                       int& /*out*/ center_index);

}   // end namespace
