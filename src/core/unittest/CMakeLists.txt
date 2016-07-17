add_executable(test_fft
               test_fft.cpp
               ../fft.hpp
               ../fft.cpp
               )
target_link_libraries(test_fft Boost::unit_test_framework)
add_test(NAME test_fft COMMAND test_fft)

add_executable(test_hilbert_mask
               test_hilbert_mask.cpp
               ../discrete_hilbert_mask.hpp
               )
target_link_libraries(test_hilbert_mask Boost::unit_test_framework)
add_test(NAME test_hilbert_mask COMMAND test_hilbert_mask)

add_executable(test_linalg
               test_linalg.cpp
               ../vector3.hpp
               )
target_link_libraries(test_linalg Boost::unit_test_framework)
add_test(NAME test_linalg COMMAND test_linalg)
