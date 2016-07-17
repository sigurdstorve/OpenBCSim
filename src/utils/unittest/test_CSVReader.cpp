#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_CSVReader
#include <boost/test/unit_test.hpp>
#include <sstream>
#include <stdexcept>
#include <functional>
#include "../CSVReader.hpp"

BOOST_AUTO_TEST_CASE(verify_reading_empty_fails) {
    using namespace csv;
    std::stringstream empty_ss;
    char delimiter = ';';
    BOOST_CHECK_THROW(CSVReader(empty_ss, delimiter), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(verify_basic_reading_works) {
    using namespace csv;
    std::vector<char> delimiters{' ', ';', ':'};
    for (char delimiter : delimiters) {
        std::stringstream ss;
        ss << "Signal_x" << delimiter << "Signal_y\n";
        for (int i = 0; i < 100; i++) {
            ss << i << delimiter << (i*i) << std::endl;
        }

        CSVReader csv_reader(ss, delimiter);

        BOOST_CHECK_NO_THROW(csv_reader.get_column<int>("Signal_x"));
        BOOST_CHECK_NO_THROW(csv_reader.get_column<int>("Signal_y"));
        BOOST_CHECK_THROW(csv_reader.get_column<int>("Signal_z"), std::runtime_error);

        const auto signal_x = csv_reader.get_column<int>("Signal_x");
        const auto signal_y = csv_reader.get_column<int>("Signal_y");
        BOOST_CHECK_EQUAL(signal_x.size(), signal_y.size());

        for (size_t i = 0; i < signal_x.size(); i++) {
            BOOST_CHECK_EQUAL(signal_x[i]*signal_x[i], signal_y[i]);
        }

        const auto float_signal_x = csv_reader.get_column<float>("Signal_x");
        const auto float_signal_y = csv_reader.get_column<float>("Signal_y");
        BOOST_CHECK_EQUAL(float_signal_x.size(), float_signal_y.size());
        BOOST_CHECK_EQUAL(signal_x.size(), float_signal_x.size());
        for (size_t i = 0; i < float_signal_x.size(); i++) {
            BOOST_CHECK_CLOSE(float_signal_x[i]*float_signal_x[i], float_signal_y[i], 1e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(verify_handle_empty_lines_at_end) {
    using namespace csv;
    const char delimiter = ';';

    std::stringstream ss;
    ss << "x" << delimiter << "y" << std::endl;
    ss << "1" << delimiter << "2" << std::endl;
    ss << "3" << delimiter << "4" << std::endl;
    ss << std::endl;
    CSVReader csv_reader(ss, delimiter);
    
    const auto col_x = csv_reader.get_column<int>("x");
    const auto col_y = csv_reader.get_column<int>("y");
    BOOST_CHECK_EQUAL(col_x.size(), col_y.size());
    BOOST_CHECK_EQUAL(col_x.size(), 2);
    BOOST_CHECK_EQUAL(col_x[0], 1);
    BOOST_CHECK_EQUAL(col_x[1], 3);
    BOOST_CHECK_EQUAL(col_y[0], 2);
    BOOST_CHECK_EQUAL(col_y[1], 4);
}

BOOST_AUTO_TEST_CASE(verify_handles_line_endings) {
    using namespace csv;
    const char delimiter = ';';
    
    std::stringstream ss1;
    ss1 << "columnx;columny\n";
    ss1 << "1;2\r\n";
    CSVReader csv_reader1(ss1, delimiter);
    BOOST_CHECK_NO_THROW(csv_reader1.get_column<int>("columnx"));
    BOOST_CHECK_NO_THROW(csv_reader1.get_column<int>("columny"));

    std::stringstream ss2;
    ss2 << "column x;column y\r\n";
    ss2 << "1;2\n\n";
    CSVReader csv_reader2(ss2, delimiter);
    BOOST_CHECK_NO_THROW(csv_reader2.get_column<int>("column x"));
    BOOST_CHECK_NO_THROW(csv_reader2.get_column<int>("column y"));
}