find_package(Boost REQUIRED COMPONENTS unit_test_framework)
add_executable(test_fft
               ../fft.hpp
               ../fft.cpp
               )
target_link_libraries(test_fft Boost::unit_test_framwork)
add_test(NAME test_fft COMMAND test_fft)
