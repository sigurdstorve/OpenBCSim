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
#include <vector>
#include "BCSimConfig.hpp"

namespace bcsim {

// Abstract interface to beam convolver objects.
class IBeamConvolver {
public:
    typedef std::unique_ptr<IBeamConvolver> ptr;
    
    // Factory function for creating beam convolvers.
    // num_proj_samples: Number of time-projection samples (also number of output samples)
    // excitation: Excitation signal.
    // Will return IQ data (with optional decimation?)
    static ptr Create(size_t num_proj_samples, const ExcitationSignal& excitation); 

    virtual ~IBeamConvolver() { }

    // Clears the time-projected signal in preparation for creating a new beam.
    // Number of samples is equal to num_proj_samples used at creation.
    virtual double* get_zeroed_time_proj_signal() = 0;
    
    // Processed the time-projections into desired output format.
    // Number of samples is equal to num_proj_samples used at creation.
    virtual std::vector<bc_float> process()     = 0;
};




}   // end namespace