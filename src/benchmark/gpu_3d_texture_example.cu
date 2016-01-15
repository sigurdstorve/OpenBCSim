#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include "algorithm/cuda_helpers.h"
#include "device_launch_parameters.h"

// Sample in xy plane (at a constant z)
__global__ void tex_kernel(cudaTextureObject_t texture_obj,
                           float x_min_normalized,
                           float x_max_normalized,
                           float y_min_normalized,
                           float y_max_normalized,
                           float z_normalized,
                           int samples,     // resolution of sampling in x and y
                           float* output    // where to store output (must be size samples*samples)
                           ) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx == 0) {
        // do all work in first thread
        for (int xi = 0; xi < samples; xi++) {
            for (int yi = 0; yi < samples; yi++) {
                const auto x_normalized = x_min_normalized + xi*(x_max_normalized-x_min_normalized)/(samples-1);
                const auto y_normalized = y_min_normalized + yi*(y_max_normalized-y_min_normalized)/(samples-1);

                const auto tex_value = tex3D<float>(texture_obj, x_normalized, y_normalized, z_normalized);
                //printf("Tex3d @ (%f, %f, %f) = %f\n", x_normalized, y_normalized, z_normalized, tex_value);
                output[xi + samples*yi] = tex_value;
            }
        }
    }
}

int main() {
    std::cout << "Demo: use CUDA 3D textures for linear interpolation" << std::endl;
    const size_t num_samples_x = 32;   // aka. width
    const size_t num_samples_y = 32;   // aka. height
    const size_t num_samples_z = 1024;  // aka. depth

    const auto x_min = -2e-2;
    const auto x_max = 2e-2;
    const auto y_min = -2e-2;
    const auto y_max = 2e-2;
    const auto z_min = 0.0;
    const auto z_max = 8e-2;

    // radius
    const auto r_min = 5e-3;        // at z_min
    const auto r_max = 13e-3;       // at z_max

    // create data to be interpolated - Gaussian cylinder
    std::vector<float> host_input_buffer;
    const auto num_total = num_samples_x*num_samples_y*num_samples_z;
    host_input_buffer.reserve(num_total);
    for (size_t zi = 0; zi < num_samples_z; zi++) {
        for (size_t yi = 0; yi < num_samples_y; yi++) {
            for (size_t xi = 0; xi < num_samples_x; xi++) {
                const auto x = x_min + xi*(x_max-x_min)/(num_samples_x-1);
                const auto y = y_min + yi*(y_max-y_min)/(num_samples_y-1);
                //const auto z = z_min + zi*(z_max-z_min)/(num_samples_z-1);
                const auto r = r_min + zi*(r_max-r_min)/(num_samples_z-1);
                //size_t index = num_samples_y*num_samples_z*xi + num_samples_x*yi + zi; // ????
                host_input_buffer.push_back( std::exp(-(x*x + y*y)/(r*r)) );
            }
        }
    }
    
    // allocate CUDA 3D array in device memory
    // channelDesc describes the format of the value returned when fetching the texture
    // ==> float texels (32, 0, 0, 0)
    auto channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array_3d;
    
    // Stackoverflow:
    //  Because you are allocating an array with cudaMalloc3DArray, use the width in elements.
    //  If you were using cudaMalloc3D, the extent would have a width in bytes
    //
    // Also: "The documentation says that if the transfer doesn't include cuda arrays, then the width parameter is always the width in bytes!".
    cudaExtent extent = make_cudaExtent(num_samples_x, num_samples_y, num_samples_z);
    cudaErrorCheck( cudaMalloc3DArray(&cu_array_3d, &channel_desc, extent, 0) );

    // copy input data from host to CUDA 3D array
    cudaMemcpy3DParms par_3d = {0};
    par_3d.srcPtr = make_cudaPitchedPtr(host_input_buffer.data(), num_samples_x*sizeof(float), num_samples_x, num_samples_y); 
    par_3d.dstArray = cu_array_3d;
    par_3d.extent = extent;
    par_3d.kind = cudaMemcpyHostToDevice;
    cudaErrorCheck( cudaMemcpy3D(&par_3d) );

    // specify texture
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array_3d;

    // specify texture object parameters
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = 1;
    tex_desc.filterMode = cudaFilterModeLinear;
    //tex_desc.filterMode = cudaFilterModePoint;

    // use border to pad with zeros outsize
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.addressMode[2] = cudaAddressModeBorder;
    tex_desc.readMode = cudaReadModeElementType;

    // create texture object
    cudaTextureObject_t tex_obj = 0;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);


    int samples = 2048; // sampling res in kernel
    const auto num_output_samples = samples*samples;

    // device buffer to hold output data
    const size_t num_output_bytes = sizeof(float)*num_output_samples;
    DeviceBufferRAII<float> device_output_buffer(num_output_bytes);

    // launch kernel - just one thread
    const auto z_normalized = 0.5f;
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    const auto pad = 0.25f;
    tex_kernel<<<grid, block>>>(tex_obj, 0.0f-pad, 1.0f+pad, 0.0f-pad, 1.0f+pad, z_normalized, samples, device_output_buffer.data());
    cudaErrorCheck( cudaDeviceSynchronize() );

    // copy result to host buffer
    std::vector<float> host_output_buffer(num_output_samples);
    cudaErrorCheck( cudaMemcpy(host_output_buffer.data(), device_output_buffer.data(), num_output_bytes, cudaMemcpyDeviceToHost) );

    cudaErrorCheck( cudaDestroyTextureObject(tex_obj) );
    cudaErrorCheck( cudaFreeArray(cu_array_3d) );

    // map to [0, 255] and store as uchar8
    std::vector<unsigned char> bytes(num_output_samples);
    for (size_t i = 0; i < num_output_samples; i++) {
        bytes[i] = static_cast<unsigned char>(host_output_buffer[i]*255);
    }

    std::ofstream out;
    out.open("d:/temp/tex3d_out.raw", std::ios::binary | std::ios::out);
    out.write(reinterpret_cast<const char*>(bytes.data()), num_output_samples);
    out.close();

    return 0;
}