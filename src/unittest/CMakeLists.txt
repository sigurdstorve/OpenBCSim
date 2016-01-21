# Handle boost
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(all_tests
               ../fft.hpp
               ../fft.cpp
               ../SimpleHDF.hpp
               ../SignalProcessing.hpp
               ../SignalProcessing.cpp
               test_linalg.cpp
               test_fft.cpp
               test_direct_conv.cpp
               test_hilbert_mask.cpp
               test_main.cpp
               )
target_link_libraries(all_tests
                      ${Boost_LIBRARIES}
                      ${HDF5_LIBRARIES}
                      )
install(TARGETS all_tests DESTINATION bin)
