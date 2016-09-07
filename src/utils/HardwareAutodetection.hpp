#pragma once
#include <string>
#include <vector>
#include "../core/export_macros.hpp"

namespace utils {

// Detect CPU and GPU hardware.
class DLL_PUBLIC HardwareAutodetector {
public:
    // Throws std::runtime_error on error.
    HardwareAutodetector();
    
    // Was the library compiled with CUDA enabled?
    bool built_with_gpu_support() const;

    // Does the system have at least one GPU CUDA device?
    bool system_has_gpu() const;

    // Returns the number of CUDA GPUs detected on the system.
    int get_num_gpus() const;

    // Was the library compiled with OpenMP support enabled?
    bool built_with_openmp_support() const;

    // Returns the maximum number of OpenMP threads the system supports.
    int max_openmp_threads() const;

    // Throws std::runtime_error on invalid device index
    std::string get_gpu_name(int device_index) const;

    // Throws std::runtime_error on invalid device index
    int get_gpu_major(int device_index) const;
    
    // Throws std::runtime_error on invalid device index
    int get_gpu_minor(int device_index) const;
    
    // Throws std::runtime_error on invalid device index
    size_t get_gpu_total_memory(int device_index) const;

private:
    bool m_built_with_gpu_support;
    bool m_system_has_gpu;
    int  m_num_gpus;
    bool m_built_with_openmp_support;
    int  m_max_openmp_threads;
    
    struct GpuDescription {
        std::string name;
        size_t      total_memory;
        int         major;
        int         minor;
    };
    std::vector<GpuDescription> m_gpu_info;
};

}   // end namespace
