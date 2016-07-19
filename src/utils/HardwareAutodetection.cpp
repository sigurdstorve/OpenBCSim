#include <stdexcept>
#ifdef BCSIM_ENABLE_CUDA
    #include <cuda_runtime_api.h>
#endif
#ifdef BCSIM_ENABLE_OPENMP
    #include <omp.h>
#endif
#include "HardwareAutodetection.hpp"

namespace utils {

void throw_if_invalid_device_index(const HardwareAutodetector& hw, int device_index) {
    if (!hw.built_with_gpu_support() || (device_index >= hw.get_num_gpus())) {
        throw std::runtime_error("Invalid GPU device number");
    }
}

HardwareAutodetector::HardwareAutodetector() {
#ifdef BCSIM_ENABLE_CUDA
    m_built_with_gpu_support = true;
    if (cudaGetDeviceCount(&m_num_gpus) != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceCount() returned failure");
    }
    for (int device_no = 0; device_no < m_num_gpus; device_no++) {
        cudaDeviceProp properties;
        if (cudaGetDeviceProperties(&properties, device_no) != cudaSuccess) {
            throw std::runtime_error("cudaGetDeviceProperties() returned failure");
        }
        GpuDescription d;
        d.major = properties.major;
        d.minor = properties.minor;
        d.name  = properties.name;
        d.total_memory = properties.totalGlobalMem;
        m_gpu_info.push_back(d);
    }
#else
    m_built_with_gpu_support = false;
#endif

#ifdef BCSIM_ENABLE_OPENMP
    m_built_with_openmp_support = true;
    m_max_openmp_threads = omp_get_max_threads();
#else
    m_built_with_openmp_support = false;
    m_max_openmp_threads = 1;
#endif
}

bool HardwareAutodetector::built_with_gpu_support() const{
    return m_built_with_gpu_support;
}

bool HardwareAutodetector::system_has_gpu() const {
    return (m_num_gpus > 0);
}

int HardwareAutodetector::get_num_gpus() const {
    return static_cast<int>(m_gpu_info.size());
}

bool HardwareAutodetector::built_with_openmp_support() const {
    return m_built_with_openmp_support;
}

int HardwareAutodetector::max_openmp_threads() const {
    return m_max_openmp_threads;
}

std::string HardwareAutodetector::get_gpu_name(int device_index) const {
    throw_if_invalid_device_index(*this, device_index);
    return m_gpu_info[device_index].name;
}

int HardwareAutodetector::get_gpu_major(int device_index) const {
    throw_if_invalid_device_index(*this, device_index);
    return m_gpu_info[device_index].major;
}

int HardwareAutodetector::get_gpu_minor(int device_index) const {
    throw_if_invalid_device_index(*this, device_index);
    return m_gpu_info[device_index].minor;
}

size_t HardwareAutodetector::get_gpu_total_memory(int device_index) const {
    throw_if_invalid_device_index(*this, device_index);
    return m_gpu_info[device_index].total_memory;
}

}   // end namespace
