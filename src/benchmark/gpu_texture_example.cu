#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "algorithm/cuda_helpers.h"
#include "device_launch_parameters.h"

__global__ void tex_kernel(cudaTextureObject_t texture_obj, int num_samples, float* output) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float u = idx / static_cast<float>(num_samples);
        output[idx] = tex1D<float>(texture_obj, u);
    }
}

int main() {
    std::cout << "Demo: use CUDA textures for linear interpolation of 1D signal" << std::endl;
    const size_t num_output_samples = 1024;
    const size_t num_input_samples = 128;
    
    // create data to be interpolated
    std::vector<float> host_input_buffer(num_input_samples);
    for (size_t i = 0; i < num_input_samples; i++) {
        host_input_buffer[i] = i*1.0f;
    }

    // allocate CUDA array in device memory
    auto channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cudaErrorCheck( cudaMallocArray(&cu_array, &channel_desc, num_input_samples) );
    const size_t num_input_bytes = sizeof(float)*num_input_samples;

    // copy input data from host to CUDA array
    cudaErrorCheck( cudaMemcpyToArray(cu_array, 0, 0, host_input_buffer.data(), num_input_bytes, cudaMemcpyHostToDevice) );


    // specify texture
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;

    // specify texture object parameters
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    // create texture object
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);

    // device buffer to hold output data
    const size_t num_output_bytes = sizeof(float)*num_output_samples;
    DeviceBufferRAII<float> device_output_buffer(num_output_bytes);

    // launch kernel
    dim3 block(128, 1, 1);
    dim3 grid(num_output_samples/block.x, 1, 1);
    tex_kernel<<<grid, block>>>(tex_obj, num_output_samples, device_output_buffer.data());
    cudaErrorCheck( cudaDeviceSynchronize() );

    // copy result to host buffer
    std::vector<float> host_output_buffer(num_output_samples);
    cudaErrorCheck( cudaMemcpy(host_output_buffer.data(), device_output_buffer.data(), num_output_bytes, cudaMemcpyDeviceToHost) );


    cudaErrorCheck( cudaDestroyTextureObject(tex_obj) );
    cudaErrorCheck( cudaFreeArray(cu_array) );

    // print results
    for (size_t i = 0; i < num_output_samples; i++) {
        std::cout << i << " : " << host_output_buffer[i] << std::endl;
    }

    return 0;
}