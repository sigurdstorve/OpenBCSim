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
#include <iostream>
#include "GpuBaseAlgorithm.cuh"

namespace bcsim {
GpuBaseAlgorithm::GpuBaseAlgorithm()
    : m_sound_speed(1540.0f),
      m_cuda_device_no(0),
      m_can_change_cuda_device(true),
      m_param_num_cuda_streams(2)
{
}

int GpuBaseAlgorithm::get_num_cuda_devices() const {
    int device_count;
    cudaErrorCheck( cudaGetDeviceCount(&device_count) );
    return device_count;
}

void GpuBaseAlgorithm::set_parameter(const std::string& key, const std::string& value) {
    if (key == "gpu_device") {
        if (!m_can_change_cuda_device) {
            throw std::runtime_error("cannot change CUDA device now");            
        }
        const auto device_count = get_num_cuda_devices();
        const int device_no = std::stoi(value);
        if (device_no < 0 || device_no >= device_count) {
            throw std::runtime_error("illegal device number");
        }
        m_cuda_device_no = device_no;
        cudaErrorCheck(cudaSetDevice(m_cuda_device_no));
        print_cuda_device_properties(m_cuda_device_no);
    } else if (key == "sound_speed") {
        m_sound_speed = std::stof(value);
    } else if (key == "cuda_streams") {
        const auto num_streams = std::stoi(value);
        if (num_streams <= 0) {
            throw std::runtime_error("invalid number of CUDA streams");
            m_param_num_cuda_streams = num_streams;
        }
    } else {
        BaseAlgorithm::set_parameter(key, value);
    }
}

void GpuBaseAlgorithm::create_cuda_stream_wrappers(int num_streams) {
    m_stream_wrappers.clear();
    for (int i = 0; i < num_streams; i++) {
        m_stream_wrappers.push_back(std::move(CudaStreamRAII::u_ptr(new CudaStreamRAII)));
    }
    m_can_change_cuda_device = false;
}

void GpuBaseAlgorithm::print_cuda_device_properties(int device_no) const {
    const auto num_devices = get_num_cuda_devices();
    if (device_no < 0 || device_no >= num_devices) {
        throw std::runtime_error("illegal CUDA device number");
    }
    cudaDeviceProp prop;
    cudaErrorCheck( cudaGetDeviceProperties(&prop, device_no) );
    std::cout << "\n\n=== Device " << device_no << ": " << prop.name               << std::endl;
    std::cout << "totalGlobMem: "               << prop.totalGlobalMem             << std::endl;
    std::cout << "clockRate: "                  << prop.clockRate                  << std::endl;
    std::cout << "Compute capability: "         << prop.major << "." << prop.minor << std::endl;
    std::cout << "asyncEngineCount: "           << prop.asyncEngineCount           << std::endl;
    std::cout << "multiProcessorCount: "        << prop.multiProcessorCount        << std::endl;
    std::cout << "kernelExecTimeoutEnabled: "   << prop.kernelExecTimeoutEnabled   << std::endl;
    std::cout << "computeMode: "                << prop.computeMode                << std::endl;
    std::cout << "concurrentKernels: "          << prop.concurrentKernels          << std::endl;
    std::cout << "ECCEnabled: "                 << prop.ECCEnabled                 << std::endl;
    std::cout << "memoryBusWidth: "             << prop.memoryBusWidth             << std::endl;
}

}   // end namespace

