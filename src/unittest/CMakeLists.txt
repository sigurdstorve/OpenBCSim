find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(all_tests
               ../utils/SignalProcessing.hpp
               ../utils/SignalProcessing.cpp
               test_linalg.cpp
               test_hilbert_mask.cpp
               test_main.cpp
               )
target_link_libraries(all_tests
                      Boost::boost
                      Boost::unit_test_framework
                      hdf5-static
                      hdf5_cpp-static
                      )
install(TARGETS all_tests DESTINATION bin)
