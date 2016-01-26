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

namespace bcsim {

// Lightweight implementation of three-dimensional vectors
// and associated operations.
template <typename T>
class Vector3D {
public:
    Vector3D () : x(static_cast<T>(0.0)),
                 y(static_cast<T>(0.0)),
                 z(static_cast<T>(0.0)) { }
    Vector3D (T x, T y, T z) : x(x), y(y), z(z) { }
    T norm() const {
        return std::sqrt(x*x+y*y+z*z);
    }
    T norm_squared() const {
        return x*x+y*y+z*z;
    }
    Vector3D<T> & operator+= (const Vector3D<T> & rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
    Vector3D<T> & operator-= (const Vector3D<T> & rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
    Vector3D<T> & operator*= (const T & rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }
    Vector3D<T> & operator/= (const T & rhs) {
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }
    const Vector3D<T> operator+ (const Vector3D<T> & rhs) const {
        return Vector3D<T>(*this) += rhs;
    }
    const Vector3D<T> operator- (const Vector3D<T> & rhs) const {
        return Vector3D<T>(*this) -= rhs;
    }
    const Vector3D<T> operator* (const T & rhs) const {
        return Vector3D<T>(*this) *= rhs;
    }
    const Vector3D<T> operator/ (const T & rhs) const {
        return Vector3D<T>(*this) /= rhs;
    }
    T dot(const Vector3D<T> & rhs) const {
        return x*rhs.x+y*rhs.y+z*rhs.z;
    }
    void normalize() {
        *this /= norm();
    }
    // Cross product of this with arg.
    Vector3D<T> cross(const Vector3D<T>& v) const {
        Vector3D<T> res;
        res.x = y*v.z - v.y*z;
        res.y = v.x*z - x*v.z;
        res.z = x*v.y - v.x*y;
        return res;
    }

public:
    T x;
    T y;
    T z;
};

// Sanity check: size of Vector3D should not be bigger than three times
// size of data type.
static_assert(sizeof(Vector3D<float>) == 3*sizeof(float), "Size of Vector<float> is not three times size of float");
static_assert(sizeof(Vector3D<double>) == 3*sizeof(double),  "Size of Vector<double> is not three times size of double");

}   // namespace

