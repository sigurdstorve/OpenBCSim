if (BCSIM_ENABLE_CUDA AND BCSIM_BUILD_UTILS)
    # Handle Boost
    # workaround for annoying program_options linking issue:
    # Setting Boost_USE_STATIC_LIBS to OFF is not enough...
    # (src http://lists.boost.org/boost-users/2009/11/54015.php)
    add_definitions(-DBOOST_ALL_DYN_LINK)
    set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "Use static linking with Boost")
    set(Boost_USE_MULTITHREADED ON CACHE BOOL "Use multithreaded Boost")
    set(Boost_USE_STATIC_RUNTIME OFF CACHE BOOL "Use static Boost runtime")
    find_package(Boost COMPONENTS program_options REQUIRED)

    # GPU algorithm example 1: fixed scatterers
    add_executable(GpuExample1 gpu_example1.cpp examples_common.hpp)
    target_link_libraries(GpuExample1
                          LibBCSim
                          BCSimCUDA
                          Boost::boost
                          Boost::program_options
                          LibBCSimUtils
                          )
    install(TARGETS GpuExample1 DESTINATION bin)
    
    # GPU algorithm example 2: spline scatterers
    add_executable(GpuExample2 gpu_example2.cpp examples_common.hpp)
    target_link_libraries(GpuExample2
                          LibBCSim
                          BCSimCUDA
                          Boost::boost
                          Boost::program_options
                          LibBCSimUtils
                          )
    install(TARGETS GpuExample2 DESTINATION bin)
endif()