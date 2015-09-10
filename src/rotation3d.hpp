/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>

// Create matrix for rotating around the x-axis.
template <typename T>
boost::numeric::ublas::matrix<T> rotation_matrix_x(T angle) {
    boost::numeric::ublas::matrix<T> m(3, 3);
    m(0,0) = static_cast<T>(1.0); m(0,1) = static_cast<T>(0.0);             m(0,2) = static_cast<T>(0.0);
    m(1,0) = static_cast<T>(0.0); m(1,1) = static_cast<T>(std::cos(angle)); m(1,2) = static_cast<T>(-std::sin(angle));
    m(2,0) = static_cast<T>(0.0); m(2,1) = static_cast<T>(std::sin(angle)); m(2,2) = static_cast<T>(std::cos(angle));
    return m;
}

// Create matrix for rotating around the y-axis
template <typename T>
boost::numeric::ublas::matrix<T> rotation_matrix_y(T angle) {
    boost::numeric::ublas::matrix<T> m(3, 3);
    m(0,0) = static_cast<T>(std::cos(angle));  m(0,1) = static_cast<T>(0.0); m(0,2) = static_cast<T>(std::sin(angle));
    m(1,0) = static_cast<T>(0.0);              m(1,1) = static_cast<T>(1.0); m(1,2) = static_cast<T>(0.0);
    m(2,0) = static_cast<T>(-std::sin(angle)); m(2,1) = static_cast<T>(0.0); m(2,2) = static_cast<T>(std::cos(angle));
    return m;
}

// Create matrix for rotating around the z-axis
template <typename T>
boost::numeric::ublas::matrix<T> rotation_matrix_z(T angle) {
    boost::numeric::ublas::matrix<T> m(3, 3);
    m(0,0) = static_cast<T>(std::cos(angle));  m(0,1) = static_cast<T>(-std::sin(angle)); m(0,2) = static_cast<T>(0.0);
    m(1,0) = static_cast<T>(std::sin(angle));  m(1,1) = static_cast<T>(std::cos(angle));  m(1,2) = static_cast<T>(0.0);
    m(2,0) = static_cast<T>(0.0);              m(2,1) = static_cast<T>(0.0);              m(2,2) = static_cast<T>(1.0);
    return m;
}

// Create matrix for rotation along x, y, z, in that order
template <typename T>
boost::numeric::ublas::matrix<T> rotation_matrix_xyz(T x_angle, T y_angle, T z_angle) {
    using namespace boost::numeric::ublas;
    const auto rot_x = rotation_matrix_x<double>(x_angle);
    const auto rot_y = rotation_matrix_y<double>(y_angle);
    const auto rot_z = rotation_matrix_z<double>(z_angle);
    boost::numeric::ublas::matrix<T> temp = prod(rot_z, rot_y);
    return prod(temp, rot_x);
}

// Return unit vector along the x-axis
template <typename T>
boost::numeric::ublas::vector<T> unit_x() {
    boost::numeric::ublas::vector<T> u(3);
    u(0) = static_cast<T>(1.0); u(1) = static_cast<T>(0.0); u(2) = static_cast<T>(0.0);
    return u;
}

// Return unit vector along the y-axis
template <typename T>
boost::numeric::ublas::vector<T> unit_y() {
    boost::numeric::ublas::vector<T> u(3);
    u(0) = static_cast<T>(0.0); u(1) = static_cast<T>(1.0); u(2) = static_cast<T>(0.0);
    return u;
}

// Return unit vector along the z-axis
template <typename T>
boost::numeric::ublas::vector<T> unit_z() {
    boost::numeric::ublas::vector<T> u(3);
    u(0) = static_cast<T>(0.0); u(1) = static_cast<T>(0.0); u(2) = static_cast<T>(1.0);
    return u;
}