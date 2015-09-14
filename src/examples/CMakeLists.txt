if (BCSIM_USE_HDF_LAYER)
    add_executable(BCSimDemo bcsim_demo.cpp)
    target_link_libraries(BCSimDemo LibBCSim)
    install(TARGETS BCSimDemo DESTINATION bin)
endif()

if (BCSIM_ENABLE_CUDA)
    add_executable(GpuExample1 gpu_example1.cpp)
    target_link_libraries(GpuExample1
                          LibBCSim
                          BCSimCUDA
                          ${Boost_LIBRARIES}
                          )
    install(TARGETS GpuExample1 DESTINATION bin)
endif()