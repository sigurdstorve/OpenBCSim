// Must do this before including unit_test.hpp
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "discrete_hilbert_mask.hpp"

bool check_equality(const std::vector<int>& mask, int* ref_mask) {
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i] != ref_mask[i]) return false;
    }
    return true;
}

BOOST_AUTO_TEST_CASE(HilbertMaskCorrectnessLength7) {
    int ref_mask[] = {1, 2, 2, 2, 0, 0, 0};
    auto mask = discrete_hilbert_mask<int>(7);
    BOOST_REQUIRE( check_equality(mask, ref_mask) );
}

BOOST_AUTO_TEST_CASE(HilbertMaskCorrectnessLength8) {
    int ref_mask[] = {1, 2, 2, 2, 1, 0, 0, 0};
    auto mask = discrete_hilbert_mask<int>(8);
    BOOST_REQUIRE( check_equality(mask, ref_mask) );
}
BOOST_AUTO_TEST_CASE(HilbertMaskCorrectnessLength11) {
    int ref_mask[] = {1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0};
    auto mask = discrete_hilbert_mask<int>(11);
    BOOST_REQUIRE( check_equality(mask, ref_mask) );
}
BOOST_AUTO_TEST_CASE(HilbertMaskCorrectnessLength18) {
    int ref_mask[] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    auto mask = discrete_hilbert_mask<int>(18);
    BOOST_REQUIRE( check_equality(mask, ref_mask) );
}