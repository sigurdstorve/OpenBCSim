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
#include <complex>
#include <vector>

// Compute forward FFT
// NOTE: THIS IS COMPLETELY UNTESTED!
// NOTE: Length must be a power of two!
template <typename T>
std::vector<std::complex<T> > fft(const std::vector<std::complex<T> >& x);
    

// Compute backward FFT (using forward FFT behind the scenes)
// NOTE: THIS IS COMPLETELY UNTESTED!
// NOTE: Length must be a power of two!
template <typename T>
std::vector<std::complex<T> > ifft(const std::vector<std::complex<T> >& x);

// Find the power of two greater than or equal to n
// This is intended to be used with fft() since it requires
// power-of-two input sizes.
size_t next_power_of_two(size_t n);

// Convolve two real signals. Output size will be Nx+Ny-1.
// Will perform zero-padding behind the scenes.
template <typename T>
std::vector<T> fft_conv(const std::vector<T>& x, const std::vector<T>& y);
