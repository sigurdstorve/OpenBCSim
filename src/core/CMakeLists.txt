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
     algorithm/CpuAlgorithm.hpp
     algorithm/CpuAlgorithm.cpp
     algorithm/common_utils.hpp
     algorithm/GpuAlgorithm.hpp
     algorithm/GpuAlgorithm.cpp
     algorithm/common_definitions.h
     algorithm/GpuScatterers.hpp
     algorithm/GpuScatterers.cpp
     )

# Made static in order to avoid DLL interface-related issues
add_library(LibBCSim STATIC ${CORE_LIBRARY_SOURCE_FILES})

if (BCSIM_ENABLE_CUDA)
    cuda_add_library(BCSimCUDA
                     algorithm/cuda_helpers.h
                     algorithm/cufft_helpers.h
                     algorithm/curand_helpers.h
                     algorithm/cuda_debug_utils.h
                     algorithm/cuda_kernels_common.cu
                     algorithm/cuda_kernels_common.cuh
                     algorithm/cuda_kernels_fixed.cu
                     algorithm/cuda_kernels_fixed.cuh
                     algorithm/cuda_kernels_spline1.cu
                     algorithm/cuda_kernels_spline1.cuh
                     algorithm/cuda_kernels_spline2.cu
                     algorithm/cuda_kernels_spline2.cuh
                     algorithm/cuda_kernels_c_interface.h
                     algorithm/cuda_kernels_c_interface.cu
                     )
    target_link_libraries(BCSimCUDA
                          ${CUDA_LIBRARIES}
                          ${CUDA_CUFFT_LIBRARIES}
                          ${CUDA_curand_LIBRARY}
                          )
    target_link_libraries(LibBCSim BCSimCUDA)
endif()

if (BCSIM_BUILD_UNITTEST)
    add_subdirectory(unittest)
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
