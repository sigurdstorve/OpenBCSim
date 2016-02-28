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

/*  This script measures the four following transfer speeds
 *  1) Pinned Host => Device
 *  2) Device => Pinned Host
 *  3) Non-Pinned Host => Device
 *  4) Device => Non-Pinned Host
 *
 *  Pinned host memory is not pageable by the OS and should therefore
 *  give better performance since no hidden intermediate copies are
 *  made.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <vector>
#include <chrono>
#include "../core/algorithm/cuda_helpers.h"

void measure_speed(size_t num_bytes, void* dst, void* src, cudaMemcpyKind kind, const std::string& msg) {
    EventTimerRAII gpu_timer;
    gpu_timer.restart();
    cudaErrorCheck( cudaMemcpy(dst, src, num_bytes, kind) );
    cudaErrorCheck( cudaDeviceSynchronize() );
    auto elapsed_ms = gpu_timer.stop();
    std::cout << "=== " << msg << " ===\n";
    std::cout << "Copied " << num_bytes << " bytes in " << elapsed_ms << "  milliseconds.\n";
    auto gb_per_sec = num_bytes/(elapsed_ms*1e6f);
    std::cout << "Transfer speed was " << gb_per_sec << " GB/sec.\n";
}

void measure_cpu_speed(size_t num_bytes) {
    std::vector<unsigned char> data(num_bytes, 1);
    const auto start_time = std::chrono::high_resolution_clock::now();
    auto copy_of_data = data;
    const auto end_time   = std::chrono::high_resolution_clock::now();
    const auto duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()*1e-3;
    const auto num_gb = num_bytes/1e9;
    std::cout << "Host => Host: " << (2.0*num_gb/duration_sec) << " GB/s \n";
}

int main() {
    const size_t num_floats = 250000000;
    const size_t num_bytes = sizeof(float)*num_floats;
    HostPinnedBufferRAII<float> pinned_host_buffer(num_bytes);
    std::cout << "Allocated " << num_bytes << " bytes of pinned host memory.\n";

    std::vector<float> regular_host_buffer(num_floats);
    std::cout << "Allocated " << num_bytes << " bytes of regular host memory.\n";
    
    DeviceBufferRAII<float> device_buffer(num_bytes);
    std::cout << "Allocated " << num_bytes << " bytes of device memory.\n";

    DeviceBufferRAII<float> device_buffer2(num_bytes);
    std::cout << "Allocated " << num_bytes << " bytes of device memory.\n";

    measure_speed(num_bytes, device_buffer.data(), pinned_host_buffer.data(), cudaMemcpyHostToDevice, "Pinned Host => Device");
    measure_speed(num_bytes, pinned_host_buffer.data(), device_buffer.data(), cudaMemcpyDeviceToHost, "Device => Pinned Host"); 
    measure_speed(num_bytes, device_buffer.data(), regular_host_buffer.data(), cudaMemcpyHostToDevice, "Regular Host => Device");
    measure_speed(num_bytes, regular_host_buffer.data(), device_buffer.data(), cudaMemcpyDeviceToHost, "Device => Regular Host");
    measure_speed(num_bytes, device_buffer.data(), device_buffer2.data(), cudaMemcpyDeviceToDevice, "Device => Device");
    std::cout << "NOTE: For Device => Device, the actual bandwidth is twice since each bytes was both read and written.\n";

    std::cout << "\nCPU memory:\n";
    measure_cpu_speed(num_bytes);
}