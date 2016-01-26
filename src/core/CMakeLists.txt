# Build core simulator library.
set(CORE_LIBRARY_SOURCE_FILES "")
list(APPEND CORE_LIBRARY_SOURCE_FILES
     BCSimConfig.hpp
     BeamConvolver.hpp
     BeamConvolver.cpp
     BeamProfile.hpp
     BeamProfile.cpp
     bspline.hpp
     discrete_hilbert_mask.hpp
     export_macros.hpp
     fft.cpp
     fft.hpp
     LibBCSim.hpp
     LibBCSim.cpp
     ScanSequence.hpp
     ScanSequence.cpp
     to_string.hpp
     to_string.cpp
     vector3.hpp
     algorithm/BaseAlgorithm.hpp
     algorithm/BaseAlgorithm.cpp
     algorithm/CpuBaseAlgorithm.hpp
     algorithm/CpuBaseAlgorithm.cpp
     algorithm/CpuFixedAlgorithm.hpp
     algorithm/CpuFixedAlgorithm.cpp
     algorithm/CpuSplineAlgorithm.hpp
     algorithm/CpuSplineAlgorithm.cpp
     algorithm/common_utils.hpp
     )

# Made static in order to avoid DLL interface-related issues
add_library(LibBCSim STATIC ${CORE_LIBRARY_SOURCE_FILES})

if (BCSIM_ENABLE_CUDA)
    if(UNIX)
        # Tell host compiler to use -fPIC
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -std=c++11 -Xcompiler -fPIC)
    else()
        # C++11 is enabled by default in MSVC
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
    endif()
    cuda_add_library(BCSimCUDA
                     algorithm/GpuBaseAlgorithm.cuh
                     algorithm/GpuBaseAlgorithm.cu
                     algorithm/gpu_alg_common.cuh
                     algorithm/gpu_alg_common.cu
                     algorithm/GpuFixedAlgorithm.cu
                     algorithm/GpuFixedAlgorithm.cuh
                     algorithm/GpuSplineAlgorithm1.cuh
                     algorithm/GpuSplineAlgorithm1.cu
                     algorithm/GpuSplineAlgorithm2.cu
                     algorithm/GpuSplineAlgorithm2.cuh
                     algorithm/cuda_helpers.h
                     algorithm/cufft_helpers.h
                     algorithm/cuda_debug_utils.h
                     )
    target_link_libraries(BCSimCUDA
                          ${CUDA_LIBRARIES}
                          ${CUDA_CUFFT_LIBRARIES}
                          )
    target_link_libraries(LibBCSim BCSimCUDA)
endif()

# Define installation of binaries and public API headers
install(TARGETS LibBCSim DESTINATION lib)

install(FILES BeamProfile.hpp      DESTINATION include)
install(FILES BCSimConfig.hpp      DESTINATION include)
install(FILES export_macros.hpp    DESTINATION include)
install(FILES LibBCSim.hpp         DESTINATION include)
install(FILES ScanSequence.hpp     DESTINATION include)
install(FILES to_string.hpp        DESTINATION include)
install(FILES vector3.hpp          DESTINATION include)
