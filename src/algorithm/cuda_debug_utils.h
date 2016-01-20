#pragma once
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda.h>
#include "cuda_helpers.h"

template <typename T>
void dump_device_buffer_as_raw_file(DeviceBufferRAII<T>& device_buffer, const std::string& raw_file, bool normalize_uchar8=true) {
    const auto num_bytes = device_buffer.get_num_bytes();
    const auto num_elements = num_bytes/sizeof(T);
    std::vector<T> host_buffer(num_elements);
    cudaErrorCheck( cudaMemcpy(host_buffer.data(), device_buffer.data(), num_bytes, cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaDeviceSynchronize() );

    std::ofstream out_stream;
    out_stream.open(raw_file, std::ios::binary | std::ios::out);

    if (normalize_uchar8) {
        const auto min_val = *std::min_element(std::begin(host_buffer), std::end(host_buffer));
        const auto max_val = *std::max_element(std::begin(host_buffer), std::end(host_buffer));

        std::vector<unsigned char> temp(num_elements);
        std::transform(std::begin(host_buffer), std::end(host_buffer), std::begin(temp), [=](float v) {
            return static_cast<unsigned char>(255.0*(v-min_val)/(max_val-min_val));
        });
        out_stream.write(reinterpret_cast<const char*>(temp.data()), num_elements*sizeof(unsigned char));

    } else {
        out_stream.write(reinterpret_cast<const char*>(host_buffer.data()), num_bytes);
    }

    out_stream.close();
    std::cout << "Wrote RAW file to " << raw_file << std::endl;
}