if (BCSIM_ENABLE_CUDA)
    cuda_add_executable(gpu_render_spline_comparison
                        gpu_render_spline_comparison.cu
                        ../algorithm/cuda_helpers.h
                        ../bspline.hpp
                        )
    target_link_libraries(gpu_render_spline_comparison
                          ${CUDA_LIBRARIES}
                          )
endif()
