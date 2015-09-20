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
#include <memory>
#include <cufft.h>
#include "cuda_helpers.h"
#include "BaseAlgorithm.hpp"

namespace bcsim {

class GpuBaseAlgorithm : public BaseAlgorithm {
public:
    GpuBaseAlgorithm();
    
    virtual void set_parameter(const std::string& key, const std::string& value);

protected:
    void create_cuda_stream_wrappers(int num_streams);
    
    int get_num_cuda_devices() const;
    
    void print_cuda_device_properties(int device_no) const;
    
protected:
    typedef cufftComplex complex;
    
    std::vector<CudaStreamRAII::u_ptr>  m_stream_wrappers;
    
    // it is only possible to change CUDA device before any operations
    // that involve the GPU
    bool        m_can_change_cuda_device;
    
    // parameters that are comon to all GPU algorithms
    float       m_sound_speed;
    int         m_cuda_device_no;
};
    
}   // end namespace

