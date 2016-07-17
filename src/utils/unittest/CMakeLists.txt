add_executable(test_SignalProcessing
    ../SignalProcessing.hpp
    ../SignalProcessing.cpp
    test_SignalProcessing.cpp
    )
target_link_libraries(test_SignalProcessing Boost::unit_test_framework)
add_test(NAME test_SignalProcessing COMMAND test_SignalProcessing)

add_executable(test_CSVReader
    ../CSVReader.hpp
    ../CSVReader.cpp
    test_CSVReader.cpp
    )
target_link_libraries(test_CSVReader Boost::unit_test_framework)
add_test(NAME test_CSVReader COMMAND test_CSVReader)
