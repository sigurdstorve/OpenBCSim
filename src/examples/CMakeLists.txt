if (BCSIM_USE_HDF_LAYER)
    add_executable(BCSimDemo bcsim_demo.cpp)
    target_link_libraries(BCSimDemo LibBCSim)
    install(TARGETS BCSimDemo DESTINATION bin)
endif()

if (BCSIM_ENABLE_CUDA)
    # GPU algorithm example 1: fixed scatterers
    add_executable(GpuExample1 gpu_example1.cpp examples_common.hpp)
    target_link_libraries(GpuExample1
                          LibBCSim
                          BCSimCUDA
                          ${Boost_LIBRARIES}
                          )
    install(TARGETS GpuExample1 DESTINATION bin)
    
    # GPU algorithm example 2: spline scatterers
    add_executable(GpuExample2 gpu_example2.cpp examples_common.hpp)
    target_link_libraries(GpuExample2
                          LibBCSim
                          BCSimCUDA
                          ${Boost_LIBRARIES}
                          )
    install(TARGETS GpuExample2 DESTINATION bin)
endif()