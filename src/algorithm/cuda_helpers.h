#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <memory>
#include <random>
#include <driver_types.h>
#include <driver_functions.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>   // for make_float3() etc.

// Throws a std::runtime_error in case the return value is not cudaSuccess.
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        auto msg = std::string("CUDA error: ")
                    + std::string(cudaGetErrorString(code))
                    + std::string(", FILE: ")
                    + std::string(file)
                    + std::string(", LINE: ")
                    + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

// RAII-style wrapper for device memory.
template <typename T>
class DeviceBufferRAII {
public:
    typedef std::unique_ptr<DeviceBufferRAII<T> > u_ptr;
    typedef std::shared_ptr<DeviceBufferRAII<T> > s_ptr;

    explicit DeviceBufferRAII(size_t num_bytes) {
        cudaErrorCheck( cudaMalloc(&memory, num_bytes) );
        num_bytes_allocated = num_bytes;
    }

    ~DeviceBufferRAII() {
        cudaErrorCheck( cudaFree(memory) );
    }

    T* data() {
        return static_cast<T*>(memory);
    }

    size_t get_num_bytes() {
        return num_bytes_allocated;
    }

private:
    void*   memory;
    size_t  num_bytes_allocated;
};

// RAII wrapper for pinned host memory.
template <typename T>
class HostPinnedBufferRAII {
public:
    typedef std::unique_ptr<HostPinnedBufferRAII<T> > u_ptr;
    typedef std::shared_ptr<HostPinnedBufferRAII<T> > s_ptr;

    explicit HostPinnedBufferRAII(size_t num_bytes) {
        cudaErrorCheck( cudaMallocHost(&memory, num_bytes) );
    }

    ~HostPinnedBufferRAII() {
        cudaErrorCheck( cudaFreeHost(memory) );
    }

    T* data() {
        return static_cast<T*>(memory);
    }
private:
    void* memory;
};

// RAII-style CUDA timer.
class EventTimerRAII {
public:
    explicit EventTimerRAII(cudaStream_t cuda_stream = 0)
        : cuda_stream(cuda_stream)
    {
        cudaErrorCheck( cudaEventCreate(&begin_event) );
        cudaErrorCheck( cudaEventCreate(&end_event) );
    }

    ~EventTimerRAII() {
        cudaErrorCheck( cudaEventDestroy(begin_event) );
        cudaErrorCheck( cudaEventDestroy(end_event) );
    }

    // restart the timer
    void restart() {
        cudaErrorCheck( cudaEventRecord(begin_event, cuda_stream) );
    }

    // return milliseconds since start
    float stop() {
        float res_millisec;
        cudaErrorCheck( cudaEventRecord(end_event, cuda_stream) );
        cudaErrorCheck( cudaEventSynchronize(end_event) );
        cudaErrorCheck( cudaEventElapsedTime(&res_millisec, begin_event, end_event) );
        
        return res_millisec;
    }

private:
    cudaEvent_t  begin_event;
    cudaEvent_t  end_event;
    cudaStream_t cuda_stream;
};

class CudaStreamRAII {
public:
    typedef std::unique_ptr<CudaStreamRAII> u_ptr;
    typedef std::shared_ptr<CudaStreamRAII> s_ptr;

    explicit CudaStreamRAII() {
        cudaErrorCheck( cudaStreamCreate(&stream) );
    }

    ~CudaStreamRAII() {
        cudaErrorCheck( cudaStreamDestroy(stream) );
    }

    cudaStream_t get() {
        return stream;
    }

private:
    cudaStream_t stream;
};

// selected math
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

// Complex datatype
typedef float2 Complex;
__device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s*a.x;
    c.y = s*a.y;
    return c;
}

__device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x*b.x - a.y*b.y;
    c.y = a.x*b.y + a.y*b.x;
    return c;
}

template <typename T>
void fill_host_vector_uniform_random(T low, T high, size_t length, T* data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(low, high);
    for (size_t i = 0; i < length; i++) {
        data[i] = dis(gen);
    }
}

inline int round_up_div(int num, int den) {
    return static_cast<int>(std::ceil(static_cast<float>(num)/den));
}

// 3D texture with tri-linear interpolation.
class DeviceBeamProfileRAII {
public:
    typedef struct TableExtent3D {
        TableExtent3D() : lateral(0), elevational(0), radial(0) { }
        TableExtent3D(size_t num_samples_lat, size_t num_samples_ele, size_t num_samples_rad)
            : lateral(num_samples_lat), elevational(num_samples_ele), radial(num_samples_rad) { }
        size_t lateral;
        size_t elevational;
        size_t radial;
    } TableExtent3D;

    DeviceBeamProfileRAII(const TableExtent3D& table_extent, std::vector<float>& host_input_buffer)
        : texture_object(0)
    {
        auto channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(table_extent.lateral, table_extent.elevational, table_extent.radial);
        cudaErrorCheck( cudaMalloc3DArray(&cu_array_3d, &channel_desc, extent, 0) );
        std::cout << "DeviceBeamProfileRAII: Allocated 3D array\n";
        
        // copy input data from host to CUDA 3D array
        cudaMemcpy3DParms par_3d = {0};
        par_3d.srcPtr = make_cudaPitchedPtr(host_input_buffer.data(), table_extent.lateral*sizeof(float), table_extent.lateral, table_extent.elevational); 
        par_3d.dstArray = cu_array_3d;
        par_3d.extent = extent;
        par_3d.kind = cudaMemcpyHostToDevice;
        cudaErrorCheck( cudaMemcpy3D(&par_3d) );
        std::cout << "DeviceBeamProfileRAII: Copied memory to device\n";

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

        // use border to pad with zeros outsize
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.addressMode[2] = cudaAddressModeBorder;
        tex_desc.readMode = cudaReadModeElementType;

        cudaErrorCheck( cudaCreateTextureObject(&texture_object, &res_desc, &tex_desc, NULL) );
        std::cout << "DeviceBeamProfileRAII: Created texture object.\n";
    }

    cudaTextureObject_t get() {
        return texture_object;
    }

    ~DeviceBeamProfileRAII() {
        cudaErrorCheck( cudaDestroyTextureObject(texture_object) );
        std::cout << "DeviceBeamProfileRAII: Destroyed texture object\n";
        cudaErrorCheck( cudaFreeArray(cu_array_3d) );
        std::cout << "DeviceBeamProfileRAII: Freed 3D array\n";
    }

private:
    cudaTextureObject_t     texture_object;
    cudaArray*              cu_array_3d;
};
