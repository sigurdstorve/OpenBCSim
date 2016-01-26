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
#include "../LibBCSim.hpp"
#include "GpuBaseAlgorithm.cuh"
#include "cuda_helpers.h"
#include "cufft_helpers.h"

// NOTE: There is no support for double here!!!

namespace bcsim {

class GpuFixedAlgorithm : public GpuBaseAlgorithm {
public:
    friend class GpuSplineAlgorithm1;

    GpuFixedAlgorithm();

    virtual ~GpuFixedAlgorithm() {
        // cleanup
    }
            
    virtual void set_scatterers(Scatterers::s_ptr new_scatterers) override;
        
protected:
    void copy_scatterers_to_device(FixedScatterers::s_ptr scatterers);

    virtual void projection_kernel(int stream_no, const Scanline& scanline, int num_blocks) override;
    
protected:
    // device memory for fixed scatterers
    DeviceBufferRAII<float>::u_ptr      m_device_point_xs;
    DeviceBufferRAII<float>::u_ptr      m_device_point_ys;
    DeviceBufferRAII<float>::u_ptr      m_device_point_zs;
    DeviceBufferRAII<float>::u_ptr      m_device_point_as;
};

}   // end namespace
