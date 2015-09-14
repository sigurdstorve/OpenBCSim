if (BCSIM_USE_HDF_LAYER)
    message(STATUS "BUILDING EXAMPLES")
    add_executable(BCSimDemo bcsim_demo.cpp)
    target_link_libraries(BCSimDemo LibBCSim)
endif()

if (BCSIM_ENABLE_CUDA)
    add_executable(GpuExample1 gpu_example1.cpp)
    target_link_libraries(GpuExample1
                          LibBCSim
                          BCSimCUDA)
endif()