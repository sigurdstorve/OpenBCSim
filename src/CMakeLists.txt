# Version 2.8.11 is minimum version for auto-linking with Qt5 libs.
cmake_minimum_required(VERSION 2.8)
project(OpenBCSim)

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
option(BCSIM_BUILD_QT5_GUI
       "Build development Qt5 code (requires BCSIM_BUILD_UTILS)" OFF)
option(BCSIM_BUILD_BENCHMARK_CODE
       "Build development code used for benchmarking" OFF)

include_directories(${PROJECT_SOURCE_DIR})

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

if (UNIX)
    add_compile_options(-std=c++11 -fvisibility=hidden)
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
    set(CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=compute_30;-gencode=arch=compute_50,code=compute_50;-use_fast_math"
        CACHE STRING "Semi-colon delimit multiple arguments.")
    find_package(CUDA REQUIRED)
    message(STATUS "Found CUDA version: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Need to add -fPIC on 64-bit Linux
if (UNIX AND NOT WIN32)
    find_program(CMAKE_UNAME uname /bin /usr/bin /usr/local/bin )
    if(CMAKE_UNAME)
        exec_program(uname ARGS -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR)
        set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR} CACHE INTERNAL
            "processor type (i386 and x86_64)")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            add_compile_options(-fPIC)
        endif ()
  endif ()
endif ()

add_subdirectory(core)

if (BCSIM_BUILD_UTILS)
    add_subdirectory(utils)
endif()

if (BCSIM_BUILD_UNITTEST)
    add_subdirectory(unittest)
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

