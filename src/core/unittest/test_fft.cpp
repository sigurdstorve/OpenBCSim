#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_fft
#include <boost/test/unit_test.hpp>
#include <complex>
#include <stdexcept> 
#include <vector>
#include <iostream>
#include <string>
#include "../fft.hpp"

template <typename T> 
std::vector<std::complex<T> > make_complex(const std::vector<T>& real_values,
                                           const std::vector<T>& imag_values) {
    auto n = real_values.size();
    if (imag_values.size() != n) throw std::runtime_error("Size mismatch");
    std::vector<std::complex<T> > res;
    res.reserve(n);
    for (size_t i = 0; i < n; i++) {
        res.push_back(std::complex<T>(real_values[i], imag_values[i]));
    }
    return res;
}

template <typename T>
void check_close_complex_vector(const std::vector<std::complex<T> >& v1,
                                const std::vector<std::complex<T> >& v2) {
    auto n = v1.size();
    if (v2.size() != n) throw std::runtime_error("Size mismatch");
    for (size_t i = 0; i < n; i++) {
        const T TOLERANCE = 1e-3;
        const T real_diff = std::abs(v1[i].real()-v2[i].real()); 
        const T imag_diff = std::abs(v1[i].imag()-v2[i].imag());
        
        if (real_diff >= TOLERANCE) {
            std::cout << "real failes at " << i << " : " << v1[i].real() << " vs. " << v2[i].real() << std::endl;
        }
        if (imag_diff >= TOLERANCE) {
            std::cout << "imag fails at " << i << " : " << v1[i].imag() << " vs. " << v2[i].imag() << std::endl;
        }
        
        BOOST_REQUIRE(real_diff < TOLERANCE);
        BOOST_REQUIRE(imag_diff < TOLERANCE);
    }
}

template <typename T>
void check_close_vector(const std::vector<T>& x, const std::vector<T>& y) {
    auto n = x.size();
    if (n != y.size()) throw std::runtime_error("Size mismatch");
    for (size_t i = 0; i < n; i++) {
        BOOST_CHECK_CLOSE(x[i], y[i], 0.01f);
    }
}

BOOST_AUTO_TEST_CASE(FFT_convolve1) {
    std::vector<float> x;
    x.push_back(1.0f);
    x.push_back(2.0f);
    x.push_back(3.0f);

    std::vector<float> h;
    h.push_back(1.0f);

    std::vector<float> desired_res;
    desired_res.push_back(1.0f);
    desired_res.push_back(2.0f);
    desired_res.push_back(3.0f);

    auto res = fft_conv(x, h);
    BOOST_REQUIRE(res.size() == x.size());
    check_close_vector(res, desired_res);
}

BOOST_AUTO_TEST_CASE(FFT_convolve2) {
    std::vector<float> x;
    x.push_back(1.0f);
    x.push_back(2.0f);
    x.push_back(3.0f);
    x.push_back(-0.23f);
    x.push_back(0.001f);
    x.push_back(32.3f);
    x.push_back(4.0f);

    std::vector<float> h;
    h.push_back(-0.33f);
    h.push_back(0.9f);
    h.push_back(-0.002f);
    h.push_back(1.1f);
    h.push_back(2.3f);
    
    std::vector<float> desired_res;
    desired_res.push_back(-0.33f);
    desired_res.push_back(0.24f);
    desired_res.push_back(0.808f);
    desired_res.push_back(3.8719f);
    desired_res.push_back(4.28667f);
    desired_res.push_back(-2.75764f);
    desired_res.push_back(34.396998f);
    desired_res.push_back(3.0075f);
    desired_res.push_back(35.5243f);
    desired_res.push_back(78.69f);
    desired_res.push_back(9.2f);

    auto res = fft_conv(x, h);

    BOOST_REQUIRE(res.size() == (x.size() + h.size() - 1));

    check_close_vector(res, desired_res);
}

BOOST_AUTO_TEST_CASE(TestNextPowerOfTwo) {
    BOOST_CHECK_EQUAL(next_power_of_two(1), 1);
    BOOST_CHECK_EQUAL(next_power_of_two(2), 2);
    BOOST_CHECK_EQUAL(next_power_of_two(3), 4);
    BOOST_CHECK_EQUAL(next_power_of_two(4), 4);
    BOOST_CHECK_EQUAL(next_power_of_two(5), 8);
    BOOST_CHECK_EQUAL(next_power_of_two(7), 8);
    BOOST_CHECK_EQUAL(next_power_of_two(8), 8);
    BOOST_CHECK_EQUAL(next_power_of_two(15), 16);
    BOOST_CHECK_EQUAL(next_power_of_two(16), 16);
    BOOST_CHECK_EQUAL(next_power_of_two(32767), 32768);
    BOOST_CHECK_EQUAL(next_power_of_two(65535), 65536);
    BOOST_CHECK_EQUAL(next_power_of_two(65536), 65536);
}