#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Test_SignalProcessing
#include <boost/test/unit_test.hpp>
#include "../SignalProcessing.hpp"

BOOST_AUTO_TEST_CASE(SanityCheck_DirectConvolution1) {
    std::vector<int> v1{1, 2, 3};
    std::vector<int> v2{1};
    std::vector<int> v = direct_conv(v1, v2);
    BOOST_REQUIRE(v.size() == v1.size() + v2.size() - 1);
    for (size_t i = 0; i < v.size(); i++) {
        BOOST_REQUIRE(v[i] == v1[i]);
    }
}

BOOST_AUTO_TEST_CASE(SanityCheck_DirectConvolution2) {
    //>>> np.convolve([1,-1,0,10, 3], [3, 9, -2]  ==> array([  3,   6, -11,  32,  99,   7,  -6])
    std::vector<int> v1{1, -1, 0, 10, 3};
    std::vector<int> v2{3, 9, -2};
    std::vector<int> desired_res{3, 6, -11, 32, 99, 7, -6};

    std::vector<int> v = direct_conv(v1, v2);
    BOOST_REQUIRE(desired_res.size() == v.size());
    for (size_t i = 0; i < v.size(); i++) {
        BOOST_REQUIRE(desired_res[i] == v[i]);
    }
}
