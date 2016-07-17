// Must do this before including unit_test.hpp
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_linalg
#include <boost/test/unit_test.hpp>
#include "../vector3.hpp"

using namespace bcsim;

BOOST_AUTO_TEST_CASE(CrossProductCorrectness) {
    Vector3D<float> xAx(1.0, 0.0, 0.0);
    Vector3D<float> yAx(0.0, 1.0, 0.0);
    Vector3D<float> zAx(0.0, 0.0, 1.0);
    Vector3D<float> temp;
    
    // x-axis cross y-axis = z-axis
    temp = xAx.cross(yAx);
    BOOST_CHECK_CLOSE(temp.x, 0.0f, 0.001f);
    BOOST_CHECK_CLOSE(temp.y, 0.0f, 0.001f);
    BOOST_CHECK_CLOSE(temp.z, 1.0f, 0.001f);
    
    // z-axis cross x-axis = y-axis
    temp = zAx.cross(xAx);
    BOOST_CHECK_CLOSE(temp.x, 0.0f, 0.001f);
    BOOST_CHECK_CLOSE(temp.y, 1.0f, 0.001f);
    BOOST_CHECK_CLOSE(temp.z, 0.0f, 0.001f);
    
    Vector3D<float> v1(1.2f, -0.32f, 9.1f);
    Vector3D<float> v2(0.2f, 0.54f, -1.13f);
    temp = v1.cross(v2);
    BOOST_CHECK_CLOSE(temp.x, -4.5524f, 0.001f);
    BOOST_CHECK_CLOSE(temp.y, 3.1760f, 0.001f);
    BOOST_CHECK_CLOSE(temp.z, 0.7120f, 0.001f);
}

BOOST_AUTO_TEST_CASE(DotProductCorrectness) {
    Vector3D<float> xAx(1.0f, 0.0f, 0.0f);
    Vector3D<float> yAx(0.0f, 1.0f, 0.0f);
    
    // x-axis dot y-axis is zero
    float res = xAx.dot(yAx);
    BOOST_CHECK_CLOSE(res, 0.0f, 0.001f);    

    // x-axis dot x-axis is one
    res = xAx.dot(xAx);
    BOOST_CHECK_CLOSE(res, 1.0f, 0.001f);    
}


