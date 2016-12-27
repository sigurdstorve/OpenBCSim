cmake_minimum_required(VERSION 3.5)
project(OpenBCSim)
cmake_policy(VERSION 3.5)
include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs")

option(BCSIM_ENABLE_OPENMP
       "Enable OpenMP support" ON)
option(BCSIM_BUILD_UNITTEST
       "Build the main unit testing code" OFF)
option(BCSIM_BUILD_UTILS
       "Build misc. utilities" OFF) 
option(BCSIM_BUILD_EXAMPLES
       "Build C++ examples (requires BCSIM_BUILD_UTILS)" OFF)
option(BCSIM_BUILD_PYTHON_INTERFACE
       "Build Python interface" OFF)
option(BCSIM_ENABLE_NAN_CHECK
       "Enable NaN checking (for debug builds)" ON)
option(BCSIM_ENABLE_CUDA
       "Build the GPU algorithms" OFF)
option(BCSIM_BUILD_BENCHMARK_CODE
       "Build development code used for benchmarking" OFF)

# The Qt5 GUI requires the "Utils" library.
cmake_dependent_option(BCSIM_BUILD_QT5_GUI "Build interactive Qt5 GUI" OFF
                       "BCSIM_BUILD_UTILS" OFF)       
       
include_directories(${PROJECT_SOURCE_DIR})

find_package(Boost REQUIRED)

if (BCSIM_BUILD_UTILS OR BCSIM_BUILD_QT5_GUI)
    # Note: Using the new-style find_package() from HDF5 which
    # makes available imported targets.
    # Then it is no longer needed to do manually do include_directories()
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    if (NOT TARGET hdf5-shared)
        message(FATAL_ERROR "Imported target hdf5-shared is not available")
    endif()
    if (NOT TARGET hdf5_cpp-shared)
        message(FATAL_ERROR "Imported target hdf5_cpp-shared is not available")
    endif()
endif()

# set compiler defines according to configuration options
if (BCSIM_ENABLE_OPENMP)
    add_definitions(-DBCSIM_ENABLE_OPENMP)
endif()
if (BCSIM_ENABLE_NAN_CHECK)
    add_definitions(-DBCSIM_ENABLE_NAN_CHECK)
endif()
if (BCSIM_ENABLE_CUDA)
    add_definitions(-DBCSIM_ENABLE_CUDA)
endif()

# C++11 is enabled by default on recent MSVC
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if (BCSIM_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if (NOT OPENMP_FOUND)
        message(FATAL_ERROR "OpenMP not found.")
    endif ()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

if (BCSIM_ENABLE_CUDA)
    # default flags for JIT compilation targeting Kepler and Maxwell, with "fast-math" enabled.
    set(CUDA_NVCC_FLAGS_DEFAULT "-gencode=arch=compute_30,code=compute_30;-gencode=arch=compute_50,code=compute_50;-use_fast_math") 
    if (CMAKE_COMPILER_IS_GNUCXX)
        # must explicitly enable C++11 with GCC. Also, position independent code is 
        # required on 64-bit systems, but should also work on 32-bit.
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_DEFAULT};-std=c++11;-Xcompiler -fPIC" CACHE STRING "")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS_DEFAULT}" CACHE STRING "")
    endif()
    find_package(CUDA REQUIRED)
    message(STATUS "Found CUDA version: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if (BCSIM_BUILD_UNITTEST)
    enable_testing()
    find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif()

add_subdirectory(core)

if (BCSIM_BUILD_UTILS)
    add_subdirectory(utils)
endif()


if (BCSIM_BUILD_EXAMPLES) 
    add_subdirectory(examples)
endif()

if (BCSIM_BUILD_PYTHON_INTERFACE)
    add_subdirectory(python)
endif()

if (BCSIM_BUILD_QT5_GUI)
    add_subdirectory(qt5gui)
endif()

if (BCSIM_BUILD_BENCHMARK_CODE)
    add_subdirectory(benchmark)
endif()

