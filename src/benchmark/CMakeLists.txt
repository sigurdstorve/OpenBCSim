if (BCSIM_ENABLE_CUDA)
    cuda_add_executable(gpu_render_spline_comparison
                        gpu_render_spline_comparison.cu
                        ../core/algorithm/cuda_helpers.h
                        ../core/bspline.hpp
                        )
    target_link_libraries(gpu_render_spline_comparison
                          ${CUDA_LIBRARIES}
                          )
    install(TARGETS gpu_render_spline_comparison DESTINATION bin)
                          
    
    cuda_add_executable(gpu_memcpy_speedtest
                        gpu_memcpy_speedtest.cu
                        ../core/algorithm/cuda_helpers.h
                        )
    target_link_libraries(gpu_memcpy_speedtest
                          ${CUDA_LIBRARIES}
                          )
    install(TARGETS gpu_memcpy_speedtest DESTINATION bin)
                          
    cuda_add_executable(gpu_texture_example
                        gpu_texture_example.cu
                        ../core/algorithm/cuda_helpers.h
                        )
    target_link_libraries(gpu_texture_example
                          ${CUDA_LIBRARIES}
                          )
    install(TARGETS gpu_texture_example DESTINATION bin)
                          
    cuda_add_executable(gpu_3d_texture_example
                        gpu_3d_texture_example.cu
                        ../core/algorithm/cuda_helpers.h
                        )
    target_link_libraries(gpu_3d_texture_example
                          ${CUDA_LIBRARIES}
                         )
    install(TARGETS gpu_3d_texture_example DESTINATION bin)
    
    cuda_add_executable(cufft_speedtest
                        cufft_speedtest.cu
                        ../core/algorithm/cuda_helpers.h
                        ../core/algorithm/cufft_helpers.h
                        )
    target_link_libraries(cufft_speedtest
                          ${CUDA_LIBRARIES}
                          ${CUDA_CUFFT_LIBRARIES}
                         )
    install(TARGETS cufft_speedtest DESTINATION bin)
endif()
