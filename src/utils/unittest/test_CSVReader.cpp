#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_CSVReader
#include <iostream> // temporary
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

        auto log_callback = [](const std::string& s) {
            std::cout << "Log message: " << s << std::endl;
        };
        CSVReader csv_reader(ss, delimiter, log_callback);

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