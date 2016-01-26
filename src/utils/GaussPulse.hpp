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

#pragma once
#include <cmath>
#include <vector>
#include "export_macros.hpp"

namespace bcsim {
/*
    Evaluates the Gaussian modulated sine wave function at one time instant
    t: time [s]
    fc: center freq [Hz]
    bw: fractional bandwidth [percent og fc]
    bwr: definition of bandwidth [dB] (typ. -6.0)
    tpr: level defined as zero [dB] (used for computing time limits)

    Relies on Gaussian Fourier transform pair:
    g(t) = np.exp(-a*t**2)
    G(f) = np.sqrt(np.pi/a)*np.exp(-np.pi**2*f**2/a)
    
*/
template <typename T>
T DLL_PUBLIC GaussianPulse(T t, T fc, T bw, T bwr);

// Return the positive time limits for when the amplitude falls below tpr (dB) in time domain.
double DLL_PUBLIC GaussianPulseTimeLimits(double fc, double bw, double bwr=-6.0, double tpr=-60.0);

// Generates a vector of samples for the times where it is non-negligible
// fc: center frequency [Hz]
// bw: fractional bandwidth
// fs: sampling frequency [Hz]
template <typename T>
void DLL_PUBLIC MakeGaussianExcitation(T fc, T bw, T fs,
                           std::vector<T>& /*out*/ times,
                           std::vector<T>& /*out*/ samples,
                           int& /*out*/ center_index);

}   // end namespace
